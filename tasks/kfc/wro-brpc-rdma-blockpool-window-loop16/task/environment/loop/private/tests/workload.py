"""Independent verifier + benchmark for the brpc RDMA block-pool / send-window module.

Every reference below was written from the module's docstrings only and in a
deliberately different style: the registration table is an interval list keyed
through the stdlib ``bisect`` instead of a scan, the pool's idle lists are
Python lists used as stacks instead of linked nodes, and the work-request
cutter is driven by a per-message cursor array instead of an emitted-slice
table.  Agreement between this reference and the module under test is therefore
evidence and not a shared bug.
"""
import bisect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("WRO_REPO_DIR", "/app/repo")
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brpc_rdma_pool_window as M  # noqa: E402

MB = 1048576
BS = (8192, 65536, 2 * MB)
NTIER = 3
MAXREG = 16
MINMB = 32
MAXMB = 1048576
MINBK = 1
MAXBK = 16
RESV = 3
MINQP = 16
MAXQP = 4096
ACKSHIFT = 1
UNSOL_BYTES = 1048576
TIERS = ("default", "large", "huge")


class Fail(AssertionError):
    pass


def _eq(what, got, want):
    if got != want:
        raise Fail("%s:\n  got  %r\n  want %r" % (what, got, want))


def _raises(what, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except ValueError:
        return
    except Fail:
        raise
    raise Fail("%s: expected ValueError, none raised" % what)


# ---------------------------------------------------------------------------
# reference: tiers and region sizing
# ---------------------------------------------------------------------------
def ref_block_type_for(size):
    if not isinstance(size, int) or isinstance(size, bool):
        raise ValueError("size")
    if size <= 0 or size > BS[-1]:
        raise ValueError("size")
    return bisect.bisect_left(BS, size)


def ref_regularize(size_mb, block_type, buckets):
    if not isinstance(size_mb, int) or isinstance(size_mb, bool):
        raise ValueError("size_mb")
    if size_mb < MINMB or size_mb > MAXMB:
        raise ValueError("size_mb")
    if block_type < 0 or block_type >= NTIER:
        raise ValueError("block_type")
    if buckets < MINBK or buckets > MAXBK:
        raise ValueError("buckets")
    raw = size_mb * MB
    step = BS[block_type] * buckets
    total = (raw // step) * step
    if total == 0:
        raise ValueError("empty region")
    return total


# ---------------------------------------------------------------------------
# reference: registration table as a bisect-keyed interval list
# ---------------------------------------------------------------------------
class RTable(object):
    def __init__(self):
        self.starts = []          # ascending
        self.rows = []            # parallel: (start, size, block_type, lkey)
        self.next_lkey = 1

    def add(self, start, size, block_type):
        if len(self.rows) >= MAXREG:
            raise ValueError("full")
        if not isinstance(start, int) or isinstance(start, bool) or start <= 0:
            raise ValueError("start")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("size")
        if start % 4096 != 0:
            raise ValueError("align")
        if block_type < 0 or block_type >= NTIER:
            raise ValueError("block_type")
        if size % BS[block_type] != 0:
            raise ValueError("size multiple")
        i = bisect.bisect_left(self.starts, start)
        if i < len(self.rows):
            s, sz, _, _ = self.rows[i]
            if s < start + size:
                raise ValueError("overlap")
        if i > 0:
            s, sz, _, _ = self.rows[i - 1]
            if start < s + sz:
                raise ValueError("overlap")
        lkey = self.next_lkey
        self.next_lkey += 1
        self.starts.insert(i, start)
        self.rows.insert(i, (start, size, block_type, lkey))
        return lkey

    def of(self, addr):
        if not isinstance(addr, int) or isinstance(addr, bool):
            raise ValueError("addr")
        i = bisect.bisect_right(self.starts, addr) - 1
        if i < 0:
            return None
        s, sz, bt, lk = self.rows[i]
        if addr < s + sz:
            return self.rows[i]
        return None

    def lkey(self, addr):
        r = self.of(addr)
        return 0 if r is None else r[3]

    def bucket(self, addr, buckets):
        if buckets < MINBK or buckets > MAXBK:
            raise ValueError("buckets")
        r = self.of(addr)
        if r is None:
            raise ValueError("unregistered")
        return ((addr - r[0]) * buckets) // r[1]


def snap_table(table):
    """Normalise the module's table to the reference's row form."""
    return [(e["start"], e["size"], e["block_type"], e["lkey"])
            for e in table["regions"]], table["next_lkey"]


def snap_rtable(rt):
    return list(rt.rows), rt.next_lkey


# ---------------------------------------------------------------------------
# reference: the tiered block pool, idle lists as Python stacks
# ---------------------------------------------------------------------------
class RPool(object):
    def __init__(self, buckets, tls_cache_num, base_addr=0x200000, seed=1):
        if not isinstance(buckets, int) or isinstance(buckets, bool):
            raise ValueError("buckets")
        if buckets < MINBK or buckets > MAXBK:
            raise ValueError("buckets")
        if not isinstance(tls_cache_num, int) or isinstance(tls_cache_num,
                                                            bool):
            raise ValueError("tls")
        if tls_cache_num < 2 or tls_cache_num > 4096 or tls_cache_num % 2:
            raise ValueError("tls")
        if (not isinstance(base_addr, int) or base_addr <= 0
                or base_addr % 4096):
            raise ValueError("base")
        self.buckets = buckets
        self.tls_cache_num = tls_cache_num
        # idle[bt][i] is a list of [start, len]; index 0 is the list head.
        self.idle = [[[] for _ in range(buckets)] for _ in range(NTIER)]
        self.exp = [[[] for _ in range(buckets)] for _ in range(NTIER)]
        self.region_num = [0] * NTIER
        self.tls = []             # index 0 is the head
        self.tls_num = 0
        self.rng = seed & 0x7FFFFFFF
        self.next_base = base_addr
        self.allocated = 0
        self.freed = 0
        self.extends = 0

    def _rand(self):
        self.rng = (self.rng * 1103515245 + 12345) & 0x7FFFFFFF
        return self.rng

    def extend(self, table, size_mb, block_type):
        if block_type < 0 or block_type >= NTIER:
            raise ValueError("block_type")
        size = ref_regularize(size_mb, block_type, self.buckets)
        base = self.next_base
        lkey = table.add(base, size, block_type)
        huge = BS[-1]
        nxt = base + size + huge
        self.next_base = -(-nxt // huge) * huge
        stripe = size // self.buckets
        for i in range(self.buckets):
            self.exp[block_type][i].insert(0, [base + i * stripe, stripe])
        self.region_num[block_type] += 1
        self.extends += 1
        return {"base": base, "size": size, "lkey": lkey, "stripe": stripe}

    def _promote(self, bt, i):
        if self.idle[bt][i]:
            raise ValueError("idle not empty")
        self.idle[bt][i] = self.exp[bt][i]
        self.exp[bt][i] = []

    def alloc(self, table, size, grow_mb=MINMB):
        bt = ref_block_type_for(size)
        bs = BS[bt]
        if bt == 0 and self.tls:
            node = self.tls.pop(0)
            self.tls_num = len(self.tls)
            self.allocated += 1
            return node[0]
        i = self._rand() % self.buckets
        if not self.idle[bt][i]:
            if self.exp[bt][i]:
                self._promote(bt, i)
            if not self.idle[bt][i]:
                self.extend(table, grow_mb, bt)
                self._promote(bt, i)
        if not self.idle[bt][i]:
            raise ValueError("still empty")
        head = self.idle[bt][i][0]
        ptr = head[0]
        if head[1] > bs:
            head[0] += bs
            head[1] -= bs
        else:
            self.idle[bt][i].pop(0)
        if bt == 0:
            lst = self.idle[0][i]
            taken = 0
            while taken < len(lst):
                if self.tls_num > self.tls_cache_num // 2:
                    break
                if lst[taken][1] > bs:
                    break
                taken += 1
                self.tls_num = taken
            if taken == 0:
                self.tls = []
            else:
                self.tls = lst[:taken]
                self.idle[0][i] = lst[taken:]
        self.allocated += 1
        return ptr

    def recycle_tls(self, table):
        moved = 0
        while self.tls:
            node = self.tls.pop(0)
            r = table.of(node[0])
            if r is None:
                continue
            i = ((node[0] - r[0]) * self.buckets) // r[1]
            self.idle[0][i].insert(0, node)
            moved += 1
        self.tls_num = 0
        return moved

    def dealloc(self, table, addr):
        r = table.of(addr)
        if r is None:
            raise ValueError("unregistered")
        bt = r[2]
        bs = BS[bt]
        if (addr - r[0]) % bs != 0:
            raise ValueError("boundary")
        node = [addr, bs]
        self.freed += 1
        if bt == 0 and self.tls_num < self.tls_cache_num:
            self.tls_num += 1
            self.tls.insert(0, node)
            return 0
        i = ((addr - r[0]) * self.buckets) // r[1]
        if bt == 0:
            num = self.tls_cache_num // 2
            batch = self.tls[:num]
            self.tls = self.tls[num:]
            if batch:
                self.idle[0][i] = batch + [node] + self.idle[0][i]
            self.tls_num -= num
        else:
            self.idle[bt][i].insert(0, node)
        return 0

    def stats(self):
        tiers = []
        for bt in range(NTIER):
            chunks = sum(len(self.idle[bt][i]) for i in range(self.buckets))
            ib = sum(n[1] for i in range(self.buckets)
                     for n in self.idle[bt][i])
            ec = sum(len(self.exp[bt][i]) for i in range(self.buckets))
            eb = sum(n[1] for i in range(self.buckets)
                     for n in self.exp[bt][i])
            mx = 0
            for i in range(self.buckets):
                for n in self.idle[bt][i]:
                    if n[1] > mx:
                        mx = n[1]
            tiers.append({"tier": TIERS[bt], "block_size": BS[bt],
                          "chunks": chunks, "idle_bytes": ib,
                          "expansion_chunks": ec, "expansion_bytes": eb,
                          "max_chunk": mx, "regions": self.region_num[bt]})
        return {"tiers": tiers, "tls_blocks": len(self.tls),
                "tls_num": self.tls_num, "allocated": self.allocated,
                "freed": self.freed, "extends": self.extends}


# ---------------------------------------------------------------------------
# reference: credit windows and the work-request cutter
# ---------------------------------------------------------------------------
def ref_window_capacities(sq, rq, rsq, rrq):
    for d in (sq, rq, rsq, rrq):
        if not isinstance(d, int) or isinstance(d, bool):
            raise ValueError("depth")
    c = [min(max(d, MINQP), MAXQP) for d in (sq, rq, rsq, rrq)]
    lc = min(c[0], c[3]) - RESV
    rc = min(c[1], c[2]) - RESV
    if lc < 1 or rc < 1:
        raise ValueError("no window")
    return {"sq_size": c[0], "rq_size": c[1], "remote_sq_size": c[2],
            "remote_rq_size": c[3], "local_cap": lc, "remote_cap": rc}


def ref_endpoint_init(sq, rq, rsq, rrq, recv_block, max_sge):
    if recv_block not in BS:
        raise ValueError("recv_block")
    if not isinstance(max_sge, int) or isinstance(max_sge, bool):
        raise ValueError("max_sge")
    if max_sge < 1 or max_sge > 64:
        raise ValueError("max_sge")
    caps = ref_window_capacities(sq, rq, rsq, rrq)
    st = dict(caps)
    st.update({"remote_recv_block": recv_block, "max_sge": max_sge,
               "remote_rq_window": caps["local_cap"],
               "sq_window": caps["local_cap"], "sq_imm_window": RESV,
               "sq_current": 0, "sq_unsignaled": 0, "unsolicited": 0,
               "unsolicited_bytes": 0, "accumulated_ack": 0, "new_rq_wrs": 0,
               "posted": 0, "imm_sent": 0, "acks_flushed": 0})
    return st


def ref_check_msgs(msgs):
    if not isinstance(msgs, (list, tuple)) or len(msgs) == 0:
        raise ValueError("msgs")
    for m in msgs:
        if not isinstance(m, dict):
            raise ValueError("msg")
        for k in ("addr", "len", "block"):
            if k not in m:
                raise ValueError("key")
            if not isinstance(m[k], int) or isinstance(m[k], bool):
                raise ValueError("int")
        if m["len"] <= 0 or m["block"] not in BS:
            raise ValueError("shape")
        if m["addr"] % m["block"] != 0 or m["len"] > m["block"]:
            raise ValueError("shape")


def ref_plan_send_wrs(st, table, msgs):
    if not isinstance(st, dict) or "sq_window" not in st:
        raise ValueError("state")
    ref_check_msgs(msgs)
    n = len(msgs)
    recv_block = st["remote_recv_block"]
    max_sge = st["max_sge"]
    lcap = st["local_cap"]
    rcap = st["remote_cap"]
    offs = [0] * n
    wrs = []
    total = 0
    cur = 0
    rrw = st["remote_rq_window"]
    sqw = st["sq_window"]
    while cur < n:
        if rrw == 0 or sqw == 0:
            if total > 0:
                break
            return {"wrs": wrs, "total_len": 0, "eagain": True,
                    "consumed": 0, "messages": n}
        slot = st["sq_current"]
        sges = []
        this_len = 0
        while True:
            if len(sges) >= max_sge or this_len >= recv_block:
                break
            if cur < n and offs[cur] >= msgs[cur]["len"]:
                cur += 1
                if cur == n:
                    break
                continue
            m = msgs[cur]
            off = offs[cur]
            take = m["len"] - off
            room = m["block"] - (off % m["block"])
            if room < take:
                take = room
            if recv_block - this_len < take:
                take = recv_block - this_len
            addr = m["addr"] + off
            lk = table.lkey(addr)
            if lk == 0:
                raise ValueError("unregistered message")
            sges.append((addr, take, lk))
            this_len += take
            offs[cur] += take
        if this_len == 0:
            continue
        imm = st["new_rq_wrs"]
        st["new_rq_wrs"] = 0
        has_tail = cur + 1 < n
        solicited = False
        if rrw == 1 or sqw == 1 or not has_tail:
            solicited = True
        elif st["unsolicited"] > lcap // 4:
            solicited = True
        elif st["accumulated_ack"] > rcap // 4:
            solicited = True
        elif st["unsolicited_bytes"] > UNSOL_BYTES:
            solicited = True
        else:
            st["unsolicited"] += 1
            st["unsolicited_bytes"] += this_len
            st["accumulated_ack"] += imm
        if solicited:
            st["unsolicited"] = 0
            st["unsolicited_bytes"] = 0
            st["accumulated_ack"] = 0
        st["sq_unsignaled"] += 1
        signaled = False
        wr_id = 0
        if st["sq_unsignaled"] >= lcap // 4:
            signaled = True
            wr_id = st["sq_unsignaled"]
            st["sq_unsignaled"] = 0
        wrs.append({"slot": slot, "num_sge": len(sges), "bytes": this_len,
                    "imm": imm, "solicited": solicited, "signaled": signaled,
                    "wr_id": wr_id, "sges": sges})
        total += this_len
        st["posted"] += 1
        st["sq_current"] += 1
        if st["sq_current"] == st["sq_size"] - RESV:
            st["sq_current"] = 0
        rrw -= 1
        sqw -= 1
    st["remote_rq_window"] = rrw
    st["sq_window"] = sqw
    return {"wrs": wrs, "total_len": total, "eagain": False,
            "consumed": sum(offs), "messages": n}


def ref_send_imm(st, imm):
    if not isinstance(imm, int) or isinstance(imm, bool) or imm < 0:
        raise ValueError("imm")
    if imm == 0 or st["sq_imm_window"] <= 0:
        return 0
    st["sq_imm_window"] -= 1
    st["imm_sent"] += 1
    return 1


def ref_send_ack(st, num):
    if not isinstance(num, int) or isinstance(num, bool) or num < 0:
        raise ValueError("num")
    old = st["new_rq_wrs"]
    st["new_rq_wrs"] = old + num
    if old > (st["remote_cap"] >> ACKSHIFT) and st["sq_imm_window"] > 0:
        imm = st["new_rq_wrs"]
        st["new_rq_wrs"] = 0
        st["acks_flushed"] += 1
        return ref_send_imm(st, imm)
    return 0


def ref_handle_completion(st, table, events):
    if not isinstance(events, (list, tuple)):
        raise ValueError("events")
    for ev in events:
        if not isinstance(ev, dict) or "kind" not in ev:
            raise ValueError("event")
        if ev["kind"] not in ("send", "recv"):
            raise ValueError("kind")
    n_send = n_recv = 0
    credits = reposted = acks = bad = 0
    slots = set()
    for ev in events:
        if ev["kind"] == "send":
            n_send += 1
            got = ev.get("wr_id", 0)
            if not isinstance(got, int) or isinstance(got, bool) or got < 0:
                raise ValueError("wr_id")
            room = st["local_cap"] - st["sq_window"]
            if got > room:
                got = room
            st["sq_window"] += got
            credits += got
            slots.add(ev.get("slot", -1))
        else:
            n_recv += 1
            imm = ev.get("imm", 0)
            if not isinstance(imm, int) or isinstance(imm, bool) or imm < 0:
                raise ValueError("imm")
            room = st["local_cap"] - st["remote_rq_window"]
            if imm > room:
                imm = room
            st["remote_rq_window"] += imm
            if table.of(ev.get("addr", 0)) is None:
                bad += 1
            else:
                reposted += 1
                acks += ref_send_ack(st, 1)
    return {"send_events": n_send, "recv_events": n_recv,
            "credits_returned": credits, "reposted": reposted, "acks": acks,
            "bad_lkey": bad, "distinct_slots": len(slots)}


def ref_window_report(wrs):
    if not isinstance(wrs, (list, tuple)):
        raise ValueError("wrs")
    if len(wrs) == 0:
        return {"wr_count": 0, "total_bytes": 0, "sge_count": 0, "max_sge": 0,
                "solicited": 0, "signaled": 0, "signal_bytes": 0,
                "imm_total": 0, "distinct_slots": 0, "max_bytes": 0,
                "first_slot": -1, "last_slot": -1}
    tb = sc = ms = so = si = sb = it = mb = 0
    slots = set()
    for wr in wrs:
        tb += wr["bytes"]
        sc += wr["num_sge"]
        if wr["num_sge"] > ms:
            ms = wr["num_sge"]
        if wr["solicited"]:
            so += 1
        if wr["signaled"]:
            si += 1
            sb += wr["bytes"]
        it += wr["imm"]
        if wr["bytes"] > mb:
            mb = wr["bytes"]
        slots.add(wr["slot"])
    return {"wr_count": len(wrs), "total_bytes": tb, "sge_count": sc,
            "max_sge": ms, "solicited": so, "signaled": si,
            "signal_bytes": sb, "imm_total": it, "distinct_slots": len(slots),
            "max_bytes": mb, "first_slot": wrs[0]["slot"],
            "last_slot": wrs[-1]["slot"]}


# ---------------------------------------------------------------------------
# reference: the whole tick
# ---------------------------------------------------------------------------
CFG_KEYS = ("buckets", "tls_cache_num", "pool_mb", "sq_size", "rq_size",
            "remote_sq_size", "remote_rq_size", "recv_block", "max_sge",
            "steps", "alloc_per_step", "msgs_per_step", "recv_per_step",
            "live_blocks", "max_alloc", "pre_regions")


def ref_check_cfg(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("cfg")
    for k in CFG_KEYS:
        if k not in cfg:
            raise ValueError("missing")
        v = cfg[k]
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("int")
    if cfg["steps"] < 1:
        raise ValueError("steps")
    for k in ("alloc_per_step", "msgs_per_step", "recv_per_step", "max_alloc"):
        if cfg[k] < 1:
            raise ValueError(k)
    if cfg["live_blocks"] < cfg["alloc_per_step"]:
        raise ValueError("live_blocks")
    if cfg["max_alloc"] > BS[-1]:
        raise ValueError("max_alloc")
    if cfg["pre_regions"] < 0 or cfg["pre_regions"] > MAXREG - 1:
        raise ValueError("pre_regions")


def ref_run_transfer_tick(cfg, seed=1):
    ref_check_cfg(cfg)
    table = RTable()
    pool = RPool(cfg["buckets"], cfg["tls_cache_num"], seed=seed)
    for i in range(cfg["pre_regions"]):
        pool.extend(table, cfg["pool_mb"], i % NTIER)
    st = ref_endpoint_init(cfg["sq_size"], cfg["rq_size"],
                           cfg["remote_sq_size"], cfg["remote_rq_size"],
                           cfg["recv_block"], cfg["max_sge"])
    rng = seed & 0x7FFFFFFF
    live = []
    digest = plans = eagains = total_bytes = total_sges = 0
    for _step in range(cfg["steps"]):
        for _ in range(cfg["alloc_per_step"]):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            size = 1 + (rng % cfg["max_alloc"])
            live.append((pool.alloc(table, size, cfg["pool_mb"]), size))
        batch = live[-cfg["msgs_per_step"]:]
        msgs = [{"addr": a, "len": s, "block": BS[ref_block_type_for(s)]}
                for (a, s) in batch]
        plan = ref_plan_send_wrs(st, table, msgs)
        rep = ref_window_report(plan["wrs"])
        plans += 1
        if plan["eagain"]:
            eagains += 1
        total_bytes += rep["total_bytes"]
        total_sges += rep["sge_count"]
        events = [{"kind": "send", "wr_id": len(plan["wrs"]), "slot": 0}]
        for wr in plan["wrs"]:
            if wr["signaled"]:
                events.append({"kind": "send", "wr_id": 0,
                               "slot": wr["slot"]})
        for (a, _s) in live[-cfg["recv_per_step"]:]:
            events.append({"kind": "recv", "imm": 1, "addr": a})
        events.append({"kind": "recv", "imm": len(plan["wrs"]),
                       "addr": live[0][0]})
        comp = ref_handle_completion(st, table, events)
        while len(live) > cfg["live_blocks"]:
            a, _s = live.pop(0)
            pool.dealloc(table, a)
        digest = (digest * 1000003
                  + rep["total_bytes"] + rep["sge_count"] * 7
                  + rep["solicited"] * 11 + rep["signaled"] * 13
                  + rep["distinct_slots"] * 17 + rep["imm_total"] * 19
                  + comp["credits_returned"] * 23 + comp["reposted"] * 29
                  + comp["acks"] * 31 + comp["distinct_slots"] * 37
                  + plan["consumed"] * 41
                  + table.bucket(live[0][0], pool.buckets) * 43
                  ) & 0xFFFFFFFFFFFF
    before = pool.stats()
    moved = pool.recycle_tls(table)
    after = pool.stats()
    return {"digest": digest, "plans": plans, "eagains": eagains,
            "total_bytes": total_bytes, "total_sges": total_sges,
            "posted": st["posted"], "imm_sent": st["imm_sent"],
            "acks_flushed": st["acks_flushed"], "sq_window": st["sq_window"],
            "remote_rq_window": st["remote_rq_window"],
            "new_rq_wrs": st["new_rq_wrs"], "sq_current": st["sq_current"],
            "regions": len(table.rows), "extends": pool.extends,
            "allocated": pool.allocated, "freed": pool.freed,
            "tls_moved": moved, "live": len(live),
            "stats_before": before, "stats_after": after}


def ref_transfer_sweep(cfgs, seed=1):
    if not isinstance(cfgs, (list, tuple)) or len(cfgs) == 0:
        raise ValueError("cfgs")
    rows = []
    guard = 0
    bt = 0
    for i, cfg in enumerate(cfgs):
        out = ref_run_transfer_tick(cfg, seed=seed + i)
        rows.append(out)
        bt += out["total_bytes"]
        guard = (guard * 31 + out["digest"] + out["posted"] * 7
                 + out["regions"] * 11 + out["extends"] * 13) & 0xFFFFFFFFFFFF
    return {"rows": rows, "guard": guard, "bytes_total": bt,
            "count": len(rows)}


# ---------------------------------------------------------------------------
# correctness suite
# ---------------------------------------------------------------------------
def _norm(o):
    """Make a module result comparable: plain dicts/lists/tuples of ints."""
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(x) for x in o]
    if isinstance(o, bool):
        return o
    return o


def _check_constants(mod):
    _eq("BLOCK_SIZES", tuple(mod.BLOCK_SIZES), BS)
    _eq("BLOCK_SIZE_COUNT", mod.BLOCK_SIZE_COUNT, NTIER)
    _eq("BLOCK_DEFAULT", mod.BLOCK_DEFAULT, 0)
    _eq("BLOCK_LARGE", mod.BLOCK_LARGE, 1)
    _eq("BLOCK_HUGE", mod.BLOCK_HUGE, 2)
    _eq("BYTES_IN_MB", mod.BYTES_IN_MB, MB)
    _eq("MIN_REGIONS", mod.MIN_REGIONS, 1)
    _eq("MAX_REGIONS", mod.MAX_REGIONS, MAXREG)
    _eq("MIN_POOL_MB", mod.MIN_POOL_MB, MINMB)
    _eq("MAX_POOL_MB", mod.MAX_POOL_MB, MAXMB)
    _eq("MIN_BUCKETS", mod.MIN_BUCKETS, MINBK)
    _eq("MAX_BUCKETS", mod.MAX_BUCKETS, MAXBK)
    _eq("RESERVED_WR_NUM", mod.RESERVED_WR_NUM, RESV)
    _eq("MIN_QP_SIZE", mod.MIN_QP_SIZE, MINQP)
    _eq("MAX_QP_SIZE", mod.MAX_QP_SIZE, MAXQP)
    _eq("ACK_FLUSH_SHIFT", mod.ACK_FLUSH_SHIFT, ACKSHIFT)
    _eq("UNSOLICITED_BYTE_LIMIT", mod.UNSOLICITED_BYTE_LIMIT, UNSOL_BYTES)
    _eq("TIER_NAMES", tuple(mod.TIER_NAMES), TIERS)
    _eq("EVENT_NAMES", tuple(mod.EVENT_NAMES), ("send", "recv"))


def _check_tiers(mod):
    probes = [1, 2, 4095, 4096, 8191, 8192, 8193, 65535, 65536, 65537,
              2 * MB - 1, 2 * MB]
    for s in probes:
        _eq("block_type_for(%d)" % s, mod.block_type_for(s),
            ref_block_type_for(s))
    for s in (0, -1, 2 * MB + 1, 1 << 40):
        _raises("block_type_for(%r)" % s, mod.block_type_for, s)
    for s in (1.0, True, "8"):
        _raises("block_type_for(%r)" % (s,), mod.block_type_for, s)

    for mb in (32, 33, 48, 64, 100, 255, 1024):
        for bt in range(NTIER):
            for bk in (1, 2, 3, 4, 5, 7, 8, 16):
                try:
                    want = ref_regularize(mb, bt, bk)
                except ValueError:
                    _raises("regularize(%d,%d,%d)" % (mb, bt, bk),
                            mod.regularize_region_size, mb, bt, bk)
                    continue
                _eq("regularize(%d,%d,%d)" % (mb, bt, bk),
                    mod.regularize_region_size(mb, bt, bk), want)
    _raises("regularize small", mod.regularize_region_size, 31, 0, 4)
    _raises("regularize big", mod.regularize_region_size, MAXMB + 1, 0, 4)
    _raises("regularize float", mod.regularize_region_size, 32.0, 0, 4)
    _raises("regularize bt", mod.regularize_region_size, 32, 3, 4)
    _raises("regularize bt-", mod.regularize_region_size, 32, -1, 4)
    _raises("regularize bk0", mod.regularize_region_size, 32, 0, 0)
    _raises("regularize bk17", mod.regularize_region_size, 32, 0, 17)
    # a 32 MB region cannot hold one 2 MB block per bucket at 16 buckets
    _raises("regularize empty", mod.regularize_region_size, 32, 2, 32)


def _check_regions(mod):
    tab = mod.region_table_new()
    rt = RTable()
    _eq("empty table", snap_table(tab), snap_rtable(rt))
    # deliberately out of order, so both the insert and the overlap check work
    adds = [(0x400000, 4 * 65536, 1), (0x200000, 8 * 8192, 0),
            (0x800000, 2 * MB, 2), (0x300000, 16 * 8192, 0),
            (0x1000000, 4 * MB, 2), (0x280000, 8192, 0)]
    for (start, size, bt) in adds:
        _eq("add(%d)" % start, mod.region_table_add(tab, start, size, bt),
            rt.add(start, size, bt))
        _eq("table after %d" % start, snap_table(tab), snap_rtable(rt))
    # overlaps: exactly on, straddling either edge, fully inside
    for (start, size, bt) in [(0x400000, 8192, 0), (0x3FF000, 2 * 8192, 0),
                              (0x401000, 8192, 0), (0x200000, 8192, 0)]:
        _raises("overlap(%d,%d)" % (start, size), mod.region_table_add, tab,
                start, size, bt)
        _raises("ref overlap(%d,%d)" % (start, size), rt.add, start, size, bt)
    _raises("unaligned start", mod.region_table_add, tab, 0x2000001, 8192, 0)
    _raises("start 0", mod.region_table_add, tab, 0, 8192, 0)
    _raises("neg start", mod.region_table_add, tab, -4096, 8192, 0)
    _raises("size 0", mod.region_table_add, tab, 0x4000000, 0, 0)
    _raises("size mult", mod.region_table_add, tab, 0x4000000, 8191, 0)
    _raises("bad bt", mod.region_table_add, tab, 0x4000000, 8192, 5)
    _raises("bad table", mod.region_table_add, [], 0x4000000, 8192, 0)
    _raises("float size", mod.region_table_add, tab, 0x4000000, 8192.0, 0)

    probes = []
    for (start, size, _bt) in adds:
        probes += [start - 1, start, start + 1, start + size - 1,
                   start + size, start + size + 1]
    probes += [0, 1, 1 << 40]
    for a in probes:
        got = mod.region_of(tab, a)
        want = rt.of(a)
        if want is None:
            _eq("region_of(%d)" % a, got, None)
        else:
            _eq("region_of(%d)" % a,
                (got["start"], got["size"], got["block_type"], got["lkey"]),
                want)
        _eq("region_lkey(%d)" % a, mod.region_lkey(tab, a), rt.lkey(a))
        for bk in (1, 2, 4, 7, 16):
            try:
                want_b = rt.bucket(a, bk)
            except ValueError:
                _raises("bucket_index(%d,%d)" % (a, bk), mod.bucket_index,
                        tab, a, bk)
                continue
            _eq("bucket_index(%d,%d)" % (a, bk),
                mod.bucket_index(tab, a, bk), want_b)
    _raises("region_of bad table", mod.region_of, None, 1)
    _raises("region_of float", mod.region_of, tab, 1.5)
    _raises("bucket_index bk0", mod.bucket_index, tab, 0x200000, 0)
    _raises("bucket_index bk17", mod.bucket_index, tab, 0x200000, 17)
    # table full
    full = mod.region_table_new()
    frt = RTable()
    for i in range(MAXREG):
        base = 0x200000 + i * 4 * MB
        mod.region_table_add(full, base, 8 * 8192, 0)
        frt.add(base, 8 * 8192, 0)
    _raises("table full", mod.region_table_add, full,
            0x200000 + MAXREG * 4 * MB, 8192, 0)
    _raises("ref table full", frt.add, 0x200000 + MAXREG * 4 * MB, 8192, 0)


def _check_pool(mod):
    for (bk, tls) in [(1, 2), (4, 128), (16, 4096), (3, 10)]:
        p = mod.pool_create(bk, tls)
        _eq("pool_create %d/%d buckets" % (bk, tls), p["buckets"], bk)
        _eq("pool_create %d/%d tls" % (bk, tls), p["tls_cache_num"], tls)
        _eq("pool_create tls empty", p["tls"], None)
        _eq("pool_create counters",
            (p["allocated"], p["freed"], p["extends"], p["tls_num"]),
            (0, 0, 0, 0))
        _eq("pool_create stats", _norm(mod.pool_stats(p)),
            _norm(RPool(bk, tls).stats()))
    _raises("pool bk0", mod.pool_create, 0, 128)
    _raises("pool bk17", mod.pool_create, 17, 128)
    _raises("pool bk float", mod.pool_create, 4.0, 128)
    _raises("pool tls1", mod.pool_create, 4, 1)
    _raises("pool tls odd", mod.pool_create, 4, 127)
    _raises("pool tls big", mod.pool_create, 4, 4098)
    _raises("pool tls float", mod.pool_create, 4, 128.0)
    _raises("pool base odd", mod.pool_create, 4, 128, 0x200001)
    _raises("pool base 0", mod.pool_create, 4, 128, 0)
    _raises("pool_stats bad", mod.pool_stats, {})
    _raises("pool_extend bad pool", mod.pool_extend, {}, None, 32, 0)

    # extend / alloc / dealloc against the reference model, several shapes
    for (bk, tls, mb, nreg) in [(4, 128, 32, 3), (1, 2, 32, 2),
                                (8, 64, 48, 4), (2, 512, 64, 3)]:
        tab = mod.region_table_new()
        pool = mod.pool_create(bk, tls, seed=5)
        rtab = RTable()
        rpool = RPool(bk, tls, seed=5)
        for i in range(nreg):
            got = mod.pool_extend(pool, tab, mb, i % NTIER)
            want = rpool.extend(rtab, mb, i % NTIER)
            _eq("extend %d/%d" % (bk, i), dict(got), want)
            _eq("extend table %d/%d" % (bk, i), snap_table(tab),
                snap_rtable(rtab))
            _eq("extend stats %d/%d" % (bk, i), _norm(mod.pool_stats(pool)),
                _norm(rpool.stats()))
        rng = 99
        held = []
        for step in range(400):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            if held and rng % 5 == 0:
                a = held.pop(0)
                _eq("dealloc %d/%d" % (bk, step),
                    mod.pool_dealloc(pool, tab, a),
                    rpool.dealloc(rtab, a))
            else:
                size = 1 + (rng >> 8) % (200 if step % 3 else 90000)
                g = mod.pool_alloc(pool, tab, size, mb)
                w = rpool.alloc(rtab, size, mb)
                _eq("alloc %d/%d size %d" % (bk, step, size), g, w)
                held.append(g)
            if step % 37 == 0:
                _eq("pool stats %d/%d" % (bk, step),
                    _norm(mod.pool_stats(pool)), _norm(rpool.stats()))
                _eq("pool table %d/%d" % (bk, step), snap_table(tab),
                    snap_rtable(rtab))
        _eq("recycle %d" % bk, mod.pool_recycle_tls(pool, tab),
            rpool.recycle_tls(rtab))
        _eq("final stats %d" % bk, _norm(mod.pool_stats(pool)),
            _norm(rpool.stats()))
        _raises("dealloc unregistered %d" % bk, mod.pool_dealloc, pool, tab,
                1 << 40)
        _raises("dealloc off-boundary %d" % bk, mod.pool_dealloc, pool, tab,
                rtab.rows[0][0] + 1)
        _raises("alloc size 0 %d" % bk, mod.pool_alloc, pool, tab, 0, mb)
        _raises("alloc too big %d" % bk, mod.pool_alloc, pool, tab,
                2 * MB + 1, mb)


def _check_window(mod):
    for depths in [(128, 128, 128, 128), (16, 16, 16, 16), (8, 8, 8, 8),
                   (4096, 4096, 4096, 4096), (9000, 32, 64, 4096),
                   (64, 256, 32, 128), (256, 64, 128, 32)]:
        _eq("window%r" % (depths,), _norm(mod.window_capacities(*depths)),
            _norm(ref_window_capacities(*depths)))
    _raises("window float", mod.window_capacities, 128.0, 128, 128, 128)
    _raises("window bool", mod.window_capacities, True, 128, 128, 128)
    for depths in [(128, 128, 128, 128), (16, 32, 64, 128)]:
        for rb in BS:
            for sge in (1, 2, 31, 32, 64):
                _eq("endpoint%r/%d/%d" % (depths, rb, sge),
                    _norm(mod.endpoint_init(*(depths + (rb, sge)))),
                    _norm(ref_endpoint_init(*(depths + (rb, sge)))))
    _raises("endpoint block", mod.endpoint_init, 128, 128, 128, 128, 1234, 4)
    _raises("endpoint sge0", mod.endpoint_init, 128, 128, 128, 128, 8192, 0)
    _raises("endpoint sge65", mod.endpoint_init, 128, 128, 128, 128, 8192, 65)
    _raises("endpoint sge float", mod.endpoint_init, 128, 128, 128, 128,
            8192, 4.0)

    st = mod.endpoint_init(128, 128, 128, 128, 8192, 8)
    rst = ref_endpoint_init(128, 128, 128, 128, 8192, 8)
    for n in (0, 1, 5, 40, 80, 1, 1, 200):
        _eq("send_ack %d" % n, mod.send_ack(st, n), ref_send_ack(rst, n))
        _eq("send_ack state %d" % n, _norm(st), _norm(rst))
    for n in (0, 1, 3, 9):
        _eq("send_imm %d" % n, mod.send_imm(st, n), ref_send_imm(rst, n))
        _eq("send_imm state %d" % n, _norm(st), _norm(rst))
    _raises("send_imm neg", mod.send_imm, st, -1)
    _raises("send_imm float", mod.send_imm, st, 1.0)
    _raises("send_ack neg", mod.send_ack, st, -1)
    _raises("send_ack bool", mod.send_ack, st, True)

    _eq("window_report empty", _norm(mod.window_report([])),
        _norm(ref_window_report([])))
    _raises("window_report bad", mod.window_report, 5)


def _check_plan(mod):
    """plan_send_wrs / handle_completion over several window shapes."""
    for (depths, rb, sge, nmsg, span) in [
            ((128, 128, 128, 128), 8192, 8, 40, 200),
            ((128, 128, 128, 128), 8192, 64, 300, 100),
            ((16, 16, 16, 16), 8192, 4, 60, 3000),
            ((32, 32, 32, 32), 65536, 16, 50, 40000),
            ((64, 64, 64, 64), 2 * MB, 32, 30, 300000),
            ((128, 256, 64, 32), 8192, 1, 20, 8192)]:
        tab = mod.region_table_new()
        pool = mod.pool_create(4, 128, seed=3)
        rtab = RTable()
        rpool = RPool(4, 128, seed=3)
        for i in range(4):
            mod.pool_extend(pool, tab, 64, i % NTIER)
            rpool.extend(rtab, 64, i % NTIER)
        st = mod.endpoint_init(*(depths + (rb, sge)))
        rst = ref_endpoint_init(*(depths + (rb, sge)))
        rng = 17
        live = []
        for rnd in range(6):
            msgs = []
            for _ in range(nmsg):
                rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
                size = 1 + rng % span
                a = mod.pool_alloc(pool, tab, size, 64)
                w = rpool.alloc(rtab, size, 64)
                _eq("plan-alloc", a, w)
                live.append(a)
                msgs.append({"addr": a, "len": size,
                             "block": BS[ref_block_type_for(size)]})
            g = mod.plan_send_wrs(st, tab, msgs)
            w = ref_plan_send_wrs(rst, rtab, msgs)
            _eq("plan%r/%d" % (depths, rnd), _norm(g), _norm(w))
            _eq("plan state%r/%d" % (depths, rnd), _norm(st), _norm(rst))
            _eq("report%r/%d" % (depths, rnd),
                _norm(mod.window_report(g["wrs"])),
                _norm(ref_window_report(w["wrs"])))
            # structural invariants the contract pins
            for wr in g["wrs"]:
                if len(wr["sges"]) > st["max_sge"]:
                    raise Fail("wr has %d sges > max_sge" % len(wr["sges"]))
                if wr["bytes"] != sum(s[1] for s in wr["sges"]):
                    raise Fail("wr bytes disagree with its sges")
                if wr["num_sge"] != len(wr["sges"]):
                    raise Fail("num_sge disagrees with sges")
                for (a2, ln, lk) in wr["sges"]:
                    if lk != mod.region_lkey(tab, a2) or lk == 0:
                        raise Fail("sge lkey wrong")
                    ent = mod.region_of(tab, a2)
                    if a2 + ln > ent["start"] + ent["size"]:
                        raise Fail("sge leaves its region")
            ev = [{"kind": "send", "wr_id": len(g["wrs"]), "slot": 0}]
            for wr in g["wrs"]:
                if wr["signaled"]:
                    ev.append({"kind": "send", "wr_id": 0,
                               "slot": wr["slot"]})
            for a2 in live[-nmsg:]:
                ev.append({"kind": "recv", "imm": 1, "addr": a2})
            ev.append({"kind": "recv", "imm": 3, "addr": 1 << 40})
            _eq("completion%r/%d" % (depths, rnd),
                _norm(mod.handle_completion(st, tab, ev)),
                _norm(ref_handle_completion(rst, rtab, ev)))
            _eq("completion state%r/%d" % (depths, rnd), _norm(st),
                _norm(rst))
            while len(live) > nmsg:
                a2 = live.pop(0)
                mod.pool_dealloc(pool, tab, a2)
                rpool.dealloc(rtab, a2)
        _eq("plan pool stats%r" % (depths,), _norm(mod.pool_stats(pool)),
            _norm(rpool.stats()))

    tab = mod.region_table_new()
    pool = mod.pool_create(4, 128, seed=3)
    mod.pool_extend(pool, tab, 32, 0)
    st = mod.endpoint_init(128, 128, 128, 128, 8192, 8)
    a = mod.pool_alloc(pool, tab, 100, 32)
    _raises("plan empty msgs", mod.plan_send_wrs, st, tab, [])
    _raises("plan bad state", mod.plan_send_wrs, {}, tab,
            [{"addr": a, "len": 4, "block": 8192}])
    _raises("plan msg not dict", mod.plan_send_wrs, st, tab, [7])
    _raises("plan msg missing", mod.plan_send_wrs, st, tab, [{"addr": a}])
    _raises("plan msg float", mod.plan_send_wrs, st, tab,
            [{"addr": a, "len": 4.0, "block": 8192}])
    _raises("plan msg len0", mod.plan_send_wrs, st, tab,
            [{"addr": a, "len": 0, "block": 8192}])
    _raises("plan msg block", mod.plan_send_wrs, st, tab,
            [{"addr": a, "len": 4, "block": 1000}])
    _raises("plan msg unaligned", mod.plan_send_wrs, st, tab,
            [{"addr": a + 1, "len": 4, "block": 8192}])
    _raises("plan msg overlong", mod.plan_send_wrs, st, tab,
            [{"addr": a, "len": 8193, "block": 8192}])
    _raises("plan msg unregistered", mod.plan_send_wrs, st, tab,
            [{"addr": 1 << 40, "len": 4, "block": 8192}])
    _raises("completion not seq", mod.handle_completion, st, tab, 5)
    _raises("completion no kind", mod.handle_completion, st, tab, [{}])
    _raises("completion bad kind", mod.handle_completion, st, tab,
            [{"kind": "nope"}])
    _raises("completion bad wr_id", mod.handle_completion, st, tab,
            [{"kind": "send", "wr_id": -1}])
    _raises("completion bad imm", mod.handle_completion, st, tab,
            [{"kind": "recv", "imm": -1}])

    # eagain: a window with nothing left to give
    st2 = mod.endpoint_init(16, 16, 16, 16, 8192, 1)
    rst2 = ref_endpoint_init(16, 16, 16, 16, 8192, 1)
    msgs = [{"addr": a, "len": 8192, "block": 8192} for _ in range(40)]
    for rnd in range(4):
        _eq("eagain plan %d" % rnd, _norm(mod.plan_send_wrs(st2, tab, msgs)),
            _norm(ref_plan_send_wrs(rst2, RTABLE_FOR(mod, tab), msgs)))
        _eq("eagain state %d" % rnd, _norm(st2), _norm(rst2))


def RTABLE_FOR(mod, tab):
    """Mirror a module region table into the reference's table object."""
    rt = RTable()
    rt.rows = [(e["start"], e["size"], e["block_type"], e["lkey"])
               for e in tab["regions"]]
    rt.starts = [r[0] for r in rt.rows]
    rt.next_lkey = tab["next_lkey"]
    return rt


TICK_CFGS = [
    dict(buckets=4, tls_cache_num=128, pool_mb=32, sq_size=128, rq_size=128,
         remote_sq_size=128, remote_rq_size=128, recv_block=8192, max_sge=8,
         steps=3, alloc_per_step=40, msgs_per_step=40, recv_per_step=40,
         live_blocks=60, max_alloc=200, pre_regions=2),
    dict(buckets=1, tls_cache_num=2, pool_mb=32, sq_size=16, rq_size=16,
         remote_sq_size=16, remote_rq_size=16, recv_block=8192, max_sge=1,
         steps=4, alloc_per_step=12, msgs_per_step=30, recv_per_step=5,
         live_blocks=12, max_alloc=8192, pre_regions=0),
    dict(buckets=8, tls_cache_num=64, pool_mb=48, sq_size=64, rq_size=256,
         remote_sq_size=32, remote_rq_size=128, recv_block=65536,
         max_sge=16, steps=3, alloc_per_step=25, msgs_per_step=50,
         recv_per_step=60, live_blocks=40, max_alloc=70000, pre_regions=5),
    dict(buckets=2, tls_cache_num=512, pool_mb=64, sq_size=256, rq_size=64,
         remote_sq_size=256, remote_rq_size=64, recv_block=2 * 1048576,
         max_sge=64, steps=3, alloc_per_step=30, msgs_per_step=30,
         recv_per_step=30, live_blocks=90, max_alloc=1500000, pre_regions=9),
    dict(buckets=3, tls_cache_num=10, pool_mb=32, sq_size=128, rq_size=128,
         remote_sq_size=128, remote_rq_size=128, recv_block=8192, max_sge=3,
         steps=5, alloc_per_step=1, msgs_per_step=1, recv_per_step=1,
         live_blocks=1, max_alloc=1, pre_regions=15),
]


def _check_tick(mod):
    for i, cfg in enumerate(TICK_CFGS):
        for seed in (1, 4):
            _eq("tick %d seed %d" % (i, seed),
                _norm(mod.run_transfer_tick(cfg, seed=seed)),
                _norm(ref_run_transfer_tick(cfg, seed=seed)))
    _eq("sweep", _norm(mod.transfer_sweep(TICK_CFGS, seed=2)),
        _norm(ref_transfer_sweep(TICK_CFGS, seed=2)))
    good = dict(TICK_CFGS[0])
    _raises("cfg not dict", mod.run_transfer_tick, 5)
    for k in CFG_KEYS:
        bad = dict(good)
        del bad[k]
        _raises("cfg missing %s" % k, mod.run_transfer_tick, bad)
        bad = dict(good)
        bad[k] = 1.5
        _raises("cfg float %s" % k, mod.run_transfer_tick, bad)
        bad = dict(good)
        bad[k] = True
        _raises("cfg bool %s" % k, mod.run_transfer_tick, bad)
    for (k, v) in [("steps", 0), ("alloc_per_step", 0), ("msgs_per_step", 0),
                   ("recv_per_step", 0), ("max_alloc", 0),
                   ("max_alloc", 2 * MB + 1), ("live_blocks", 1),
                   ("pre_regions", -1), ("pre_regions", MAXREG)]:
        bad = dict(good)
        bad[k] = v
        _raises("cfg %s=%r" % (k, v), mod.run_transfer_tick, bad)
    _raises("sweep empty", mod.transfer_sweep, [])
    _raises("sweep not seq", mod.transfer_sweep, 5)


def nontrivial(mod):
    _check_constants(mod)
    _check_tiers(mod)
    _check_regions(mod)
    _check_pool(mod)
    _check_window(mod)
    _check_plan(mod)
    _check_tick(mod)
    return True


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
SCALE = int(os.environ.get("WRO_BRPC_SCALE", "1"))

BENCH_CFGS = [
    dict(buckets=4, tls_cache_num=512, pool_mb=32, sq_size=128, rq_size=128,
         remote_sq_size=128, remote_rq_size=128, recv_block=8192, max_sge=64,
         steps=12 * SCALE, alloc_per_step=640, msgs_per_step=640,
         recv_per_step=640, live_blocks=640, max_alloc=100, pre_regions=14),
    dict(buckets=8, tls_cache_num=256, pool_mb=64, sq_size=256, rq_size=256,
         remote_sq_size=256, remote_rq_size=256, recv_block=65536,
         max_sge=48, steps=4 * SCALE, alloc_per_step=480, msgs_per_step=480,
         recv_per_step=480, live_blocks=480, max_alloc=1200, pre_regions=9),
    dict(buckets=2, tls_cache_num=384, pool_mb=128, sq_size=64, rq_size=64,
         remote_sq_size=64, remote_rq_size=64, recv_block=2 * 1048576,
         max_sge=64, steps=2 * SCALE, alloc_per_step=512, msgs_per_step=512,
         recv_per_step=512, live_blocks=512, max_alloc=60000, pre_regions=6),
]


def run_bench(mod):
    out = mod.transfer_sweep(BENCH_CFGS, seed=7)
    guard = out["guard"] + out["bytes_total"] * 3 + out["count"]
    for row in out["rows"]:
        guard += (row["digest"] + row["posted"] * 5 + row["total_sges"] * 7
                  + row["allocated"] * 11 + row["freed"] * 13
                  + row["tls_moved"] * 17 + row["regions"] * 19
                  + row["imm_sent"] * 23 + row["acks_flushed"] * 29
                  + row["sq_current"] * 31 + row["eagains"] * 37)
        for st in (row["stats_before"], row["stats_after"]):
            guard += st["tls_blocks"] + st["allocated"] + st["freed"]
            for t in st["tiers"]:
                guard += (t["chunks"] + t["idle_bytes"] + t["max_chunk"]
                          + t["expansion_chunks"] + t["regions"])
    return guard & 0xFFFFFFFFFFFFFF


def main(argv):
    mode = argv[1] if len(argv) > 1 else "all"
    if mode not in ("all", "correctness", "timing"):
        sys.stderr.write("usage: workload.py [all|correctness|timing]\n")
        return 2
    ok = True
    ms = -1.0
    guard = 0
    if mode in ("all", "correctness"):
        try:
            nontrivial(M)
            print("correctness OK")
        except Exception as e:  # noqa: BLE001
            ok = False
            sys.stderr.write("CORRECTNESS FAIL: %r\n" % (e,))
            import traceback
            traceback.print_exc()
    if mode in ("all", "timing"):
        guard = run_bench(M)          # warm-up on the real, full-size inputs
        samples = []
        for _ in range(3):
            t0 = time.process_time()
            g = run_bench(M)
            samples.append((time.process_time() - t0) * 1000.0)
            if g != guard:
                ok = False
                sys.stderr.write("GUARD DRIFT %r vs %r\n" % (g, guard))
        samples.sort()
        ms = samples[1]
        print("timing_ms=%.4f guard=%d scale=%d" % (ms, guard, SCALE))
    print("WRO_BRPC_RESULT " + json.dumps({
        "correctness_ok": bool(ok),
        "timing_ms": round(ms, 4),
        "guard": int(guard),
        "scale": SCALE,
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
