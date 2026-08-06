#!/usr/bin/env python3
"""Workload + independent reference for wro-nccl-ib-multisend-wr-plan.

modes:
  correctness  -> prints WRO_IBMS_RESULT {"correctness_ok": true, ...}
  timing       -> prints WRO_IBMS_RESULT {"timing_ms": <float>, ...}

The reference below is written independently of /app/repo: it walks the queue
pairs and applies the ``len_i = min(size - off_i, len_{i-1})`` recurrence
directly, with no closed form and no shared helper.
"""
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, "/app/repo")

TOKEN = "WRO_IBMS_RESULT"
ALIGN = 128
MAX_RECVS = 8
FIFO_DEPTH = 128
U32 = 0xFFFFFFFF
REC_SIZE = MAX_RECVS * 4
BY_ID = 0
BY_INDEX = 1

COLS = ("qp_index", "dev_index", "req_index", "length", "num_sge",
        "lkey", "rkey", "laddr", "raddr", "wr_id")


# --------------------------------------------------------------------------
# independent reference
# --------------------------------------------------------------------------
def ref_span(size, nqps):
    if size == 0:
        return 0
    per = (size + nqps - 1) // nqps
    return ((per + ALIGN - 1) // ALIGN) * ALIGN


def ref_wr_ids(slot, nreqs):
    byte = slot & 0xFF
    acc = 0
    out = []
    for r in range(nreqs):
        acc = acc + (byte << (8 * r))
        out.append(acc)
    return out


def ref_ring(nqps, req_id, qp_dev, qp_remdev):
    out = []
    for i in range(nqps):
        q = (req_id + i) % nqps
        out.append((q, qp_dev[q], qp_remdev[q]))
    return out


def ref_plan(sizes, data, lkeys, raddrs, rkeys, slot, nqps, qp_dev, qp_remdev,
             req0_id):
    nreqs = len(sizes)
    wr_ids = ref_wr_ids(slot, nreqs)
    ring = ref_ring(nqps, req0_id, qp_dev, qp_remdev)
    chunks = [ref_span(sizes[r], nqps) for r in range(nreqs)]
    cur = list(chunks)
    off = [0] * nreqs
    la = list(data)
    ra = list(raddrs)
    cols = dict((k, []) for k in COLS)
    for i in range(nqps):
        q, d, rd = ring[i]
        for r in range(nreqs):
            left = sizes[r] - off[r]
            length = left if left < cur[r] else cur[r]
            cols["qp_index"].append(q)
            cols["dev_index"].append(d)
            cols["req_index"].append(r)
            cols["length"].append(length)
            cols["num_sge"].append(1 if length else 0)
            cols["lkey"].append(lkeys[r][d] if length else 0)
            cols["rkey"].append(rkeys[r][rd])
            cols["laddr"].append(la[r])
            cols["raddr"].append(ra[r])
            cols["wr_id"].append(wr_ids[r])
            cur[r] = length
            nxt = off[r] + length
            off[r] = nxt if nxt < sizes[r] else sizes[r]
            la[r] += length
            ra[r] += length
    out = {"n": nqps * nreqs, "nqps": nqps, "nreqs": nreqs,
           "opcode": "RDMA_WRITE", "chunk": chunks, "posted": list(off)}
    out.update(cols)
    return out


def ref_stats(plan):
    nqps = plan["nqps"]
    nreqs = plan["nreqs"]
    per_qp = [0] * nqps
    per_req = [0] * nreqs
    zero = 0
    sge = 0
    mx = 0
    for j in range(plan["n"]):
        L = plan["length"][j]
        per_qp[plan["qp_index"][j]] += L
        per_req[plan["req_index"][j]] += L
        if L == 0:
            zero += 1
        else:
            sge += 1
            if L > mx:
                mx = L
    return {"bytes_per_qp": per_qp, "bytes_per_req": per_req,
            "total_bytes": sum(per_req), "zero_len_wrs": zero,
            "sge_wrs": sge, "max_len": mx}


def ref_last(nreqs, req0_id, req0_size, scheme, ar, ar_thr, rem_ooo, local_ooo,
             cmpls_addr, slot, cmpls_rkeys):
    extra = bool(nreqs > 1 or ((not (rem_ooo and local_ooo)) and ar
                               and req0_size > ar_thr))
    imm = (req0_id % U32) if scheme == BY_ID else req0_size
    imm = imm & U32
    b = [(imm >> 0) & 0xFF, (imm >> 8) & 0xFF, (imm >> 16) & 0xFF, (imm >> 24) & 0xFF]
    imm_be = (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]
    out = {"extra_wr": extra, "opcode": "RDMA_WRITE_WITH_IMM", "signaled": True,
           "imm_data": imm, "imm_be": imm_be,
           "wr_id": ref_wr_ids(slot, nreqs)[-1],
           "chain_len": nreqs + (1 if extra else 0)}
    if extra and nreqs > 1:
        out["num_sge"] = 1
        out["sizes_len"] = nreqs * 4
        out["remote_addr"] = cmpls_addr + slot * REC_SIZE
        out["rkey_by_dev"] = list(cmpls_rkeys)
    else:
        out["num_sge"] = 0
        out["sizes_len"] = 0
        out["remote_addr"] = 0
        out["rkey_by_dev"] = []
    return out


def ref_match(slots, expected_idx, tag, size, taken):
    if slots[0]["idx"] != expected_idx:
        return (-1, -1)
    nreqs = slots[0]["nreqs"]
    for r in range(nreqs):
        if taken[r]:
            continue
        s = slots[r]
        if s["tag"] != tag:
            continue
        if s["size"] < 0 or s["addr"] == 0 or s["rkeys"][0] == 0:
            raise ValueError("bad slot")
        return (r, size if size < s["size"] else s["size"])
    return (-1, -1)


def ref_drain(batches, nqps, qp_dev, qp_remdev, scheme, ar, ar_thr, rem_ooo,
              local_ooo, cmpls_addr, cmpls_rkeys):
    out = []
    for b in batches:
        slots = b["slots"]
        nreqs = slots[0]["nreqs"]
        taken = [False] * nreqs
        sizes = [0] * nreqs
        data = [0] * nreqs
        lks = [None] * nreqs
        matched = 0
        for p in b["posts"]:
            r, cs = ref_match(slots, b["idx"], p["tag"], p["size"], taken)
            if r < 0:
                continue
            taken[r] = True
            sizes[r] = cs
            data[r] = p["data"]
            lks[r] = p["lkeys"]
            matched += 1
        if matched < nreqs:
            continue
        plan = ref_plan(sizes, data, lks,
                        [slots[r]["addr"] for r in range(nreqs)],
                        [slots[r]["rkeys"] for r in range(nreqs)],
                        b["slot"], nqps, qp_dev, qp_remdev, b["req0_id"])
        out.append({"slot": b["slot"], "stats": ref_stats(plan),
                    "last": ref_last(nreqs, b["req0_id"], sizes[0], scheme, ar,
                                     ar_thr, rem_ooo, local_ooo, cmpls_addr,
                                     b["slot"], cmpls_rkeys)})
    return out


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
SIZE_MIX = [0, 128, 512, 1024, 4096, 8192, 16384, 32768, 65536, 131072,
            262144, 1000, 3000, 12345, 99999]


def make_batches(n_batches, nqps, ndevs, rem_ndevs, seed):
    rng = random.Random(seed)
    batches = []
    for b in range(n_batches):
        nreqs = rng.choice([1, 1, 2, 2, 3, 4, 4, 5, 6, 8])
        idx = b + 1
        slot = b % FIFO_DEPTH
        shared_tag = (b % 8 == 3)
        if shared_tag:
            tags = [4242] * nreqs
        else:
            tags = rng.sample(range(1000, 9999), nreqs)
        slots = []
        posts = []
        for r in range(nreqs):
            size = rng.choice(SIZE_MIX)
            slack = rng.choice([0, 0, 0, 0, 512, 4096])
            slots.append({
                "idx": idx, "nreqs": nreqs, "tag": tags[r],
                "size": size + slack,
                "addr": 0x7F0000000000 + ((b * MAX_RECVS + r) << 16),
                "rkeys": [rng.randrange(1, 1 << 24) for _ in range(rem_ndevs)],
            })
            posts.append({
                "tag": tags[r], "size": size,
                "data": 0x600000000000 + ((b * MAX_RECVS + r) << 16),
                "lkeys": [rng.randrange(1, 1 << 24) for _ in range(ndevs)],
            })
        rng.shuffle(posts)
        batches.append({"idx": idx, "slot": slot, "slots": slots,
                        "posts": posts, "req0_id": b * 7 + 3})
    return batches


TIMING_NQPS = 8
TIMING_NDEVS = 2
TIMING_REMDEVS = 2
TIMING_BATCHES = int(os.environ.get("WRO_IBMS_BATCHES", "1536"))
TIMING_SEED = 20260726
TIMING_FABRIC = {
    "qp_dev": [0, 1, 0, 1, 0, 1, 0, 1],
    "qp_remdev": [1, 0, 1, 0, 0, 1, 1, 0],
    "cmpls_addr": 0x7E0000000000,
    "cmpls_rkeys": [0xABCD01, 0xABCD02],
}


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
class Fail(Exception):
    pass


def _cmp(name, got, want):
    if type(got) is not type(want):
        raise Fail("%s: type %s != %s" % (name, type(got).__name__,
                                          type(want).__name__))
    if isinstance(want, list):
        if len(got) != len(want):
            raise Fail("%s: len %d != %d" % (name, len(got), len(want)))
        for i, (g, w) in enumerate(zip(got, want)):
            _cmp("%s[%d]" % (name, i), g, w)
        return
    if isinstance(want, dict):
        if set(got) != set(want):
            raise Fail("%s: keys %s != %s" % (name, sorted(got), sorted(want)))
        for k in sorted(want):
            _cmp("%s.%s" % (name, k), got[k], want[k])
        return
    if isinstance(want, tuple):
        if len(got) != len(want):
            raise Fail("%s: tuple len %d != %d" % (name, len(got), len(want)))
        for i, (g, w) in enumerate(zip(got, want)):
            _cmp("%s[%d]" % (name, i), g, w)
        return
    if got != want:
        raise Fail("%s: %r != %r" % (name, got, want))


def _raises(name, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except ValueError:
        return
    except Fail:
        raise
    except Exception as e:  # noqa: BLE001
        raise Fail("%s: raised %s, expected ValueError" % (name, type(e).__name__))
    raise Fail("%s: did not raise ValueError" % name)


# --------------------------------------------------------------------------
# hand-traced anti-hardcode gate
# --------------------------------------------------------------------------
def nontrivial(M):
    """A 2-QP / 2-request batch whose plan was traced by hand.

    nqps=2, slot=3, req0_id=5, qp_dev=[0,1], qp_remdev=[1,0].
    req0 size 300 -> span ceil(150)->256 -> lengths [256, 44]
    req1 size  64 -> span ceil(32)->128  -> lengths [64, 0]
    QP order: (5+0)%2=1 then (5+1)%2=0.
    """
    sizes = [300, 64]
    data = [0x1000, 0x2000]
    lkeys = [[0x11, 0x12], [0x21, 0x22]]
    raddrs = [0x9000, 0xA000]
    rkeys = [[0x31, 0x32], [0x41, 0x42]]
    plan = M.plan_multisend(sizes, data, lkeys, raddrs, rkeys, 3, 2,
                            [0, 1], [1, 0], 5)
    _cmp("hand.chunk", plan["chunk"], [256, 128])
    _cmp("hand.qp_index", plan["qp_index"], [1, 1, 0, 0])
    _cmp("hand.dev_index", plan["dev_index"], [1, 1, 0, 0])
    _cmp("hand.req_index", plan["req_index"], [0, 1, 0, 1])
    _cmp("hand.length", plan["length"], [256, 64, 44, 0])
    _cmp("hand.num_sge", plan["num_sge"], [1, 1, 1, 0])
    _cmp("hand.lkey", plan["lkey"], [0x12, 0x22, 0x11, 0])
    _cmp("hand.rkey", plan["rkey"], [0x31, 0x41, 0x32, 0x42])
    _cmp("hand.laddr", plan["laddr"], [0x1000, 0x2000, 0x1100, 0x2040])
    _cmp("hand.raddr", plan["raddr"], [0x9000, 0xA000, 0x9100, 0xA040])
    _cmp("hand.wr_id", plan["wr_id"], [3, 771, 3, 771])
    _cmp("hand.posted", plan["posted"], [300, 64])
    st = M.multisend_stats(plan)
    _cmp("hand.bytes_per_qp", st["bytes_per_qp"], [44, 320])
    _cmp("hand.bytes_per_req", st["bytes_per_req"], [300, 64])
    _cmp("hand.total_bytes", st["total_bytes"], 364)
    _cmp("hand.zero_len_wrs", st["zero_len_wrs"], 1)
    _cmp("hand.sge_wrs", st["sge_wrs"], 3)
    _cmp("hand.max_len", st["max_len"], 256)
    last = M.last_wr_plan(2, 5, 300, M.MATCH_BY_INDEX, False, 1 << 20, False,
                          False, 0xB000, 3, [0x51, 0x52])
    _cmp("hand.last.extra_wr", last["extra_wr"], True)
    _cmp("hand.last.imm_data", last["imm_data"], 300)
    _cmp("hand.last.imm_be", last["imm_be"], 0x2C010000)
    _cmp("hand.last.sizes_len", last["sizes_len"], 8)
    _cmp("hand.last.remote_addr", last["remote_addr"], 0xB000 + 3 * 32)
    _cmp("hand.last.rkey_by_dev", last["rkey_by_dev"], [0x51, 0x52])
    _cmp("hand.last.chain_len", last["chain_len"], 3)
    _cmp("hand.last.wr_id", last["wr_id"], 771)


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------
def correctness():
    import ib_multisend_planner as M
    checks = 0

    nontrivial(M)
    checks += 1

    # chunk_span
    for size, nqps, want in [(0, 4, 0), (1, 1, 128), (128, 1, 128),
                             (129, 1, 256), (300, 2, 256), (64, 2, 128),
                             (1024, 8, 128), (1025, 8, 256), (99999, 8, 12544),
                             (7, 8, 128)]:
        _cmp("chunk_span(%d,%d)" % (size, nqps), M.chunk_span(size, nqps), want)
        _cmp("chunk_span.ref(%d,%d)" % (size, nqps), M.chunk_span(size, nqps),
             ref_span(size, nqps))
        checks += 1
    _raises("chunk_span.nqps0", M.chunk_span, 100, 0)
    _raises("chunk_span.neg", M.chunk_span, -1, 4)
    checks += 2

    # pack_wr_ids
    _cmp("pack(3,2)", M.pack_wr_ids(3, 2), [3, 771])
    _cmp("pack(0,1)", M.pack_wr_ids(0, 1), [0])
    _cmp("pack(255,3)", M.pack_wr_ids(255, 3), [255, 65535, 16777215])
    _cmp("pack(256,2)", M.pack_wr_ids(256, 2), [0, 0])
    _cmp("pack(127,8)", M.pack_wr_ids(127, 8), ref_wr_ids(127, 8))
    _raises("pack.slotneg", M.pack_wr_ids, -1, 2)
    _raises("pack.nreqs0", M.pack_wr_ids, 3, 0)
    _raises("pack.nreqs9", M.pack_wr_ids, 3, 9)
    checks += 8

    # qp_ring
    _cmp("ring(4,6)", M.qp_ring(4, 6, [0, 1, 2, 3], [3, 2, 1, 0]),
         [(2, 2, 1), (3, 3, 0), (0, 0, 3), (1, 1, 2)])
    _cmp("ring(1,9)", M.qp_ring(1, 9, [0], [0]), [(0, 0, 0)])
    _cmp("ring.ref", M.qp_ring(8, 13, TIMING_FABRIC["qp_dev"],
                               TIMING_FABRIC["qp_remdev"]),
         ref_ring(8, 13, TIMING_FABRIC["qp_dev"], TIMING_FABRIC["qp_remdev"]))
    _raises("ring.nqps0", M.qp_ring, 0, 1, [], [])
    _raises("ring.ragged", M.qp_ring, 4, 1, [0, 1], [0, 1, 2, 3])
    _raises("ring.negid", M.qp_ring, 4, -1, [0, 1, 2, 3], [0, 1, 2, 3])
    checks += 6

    # match_send_slot
    slots = [
        {"idx": 7, "nreqs": 3, "tag": 11, "size": 100, "addr": 0x1000, "rkeys": [5, 6]},
        {"idx": 7, "nreqs": 3, "tag": 11, "size": 40, "addr": 0x2000, "rkeys": [7, 8]},
        {"idx": 7, "nreqs": 3, "tag": 22, "size": 900, "addr": 0x3000, "rkeys": [9, 1]},
    ]
    _cmp("match.stale", M.match_send_slot(slots, 8, 11, 50, [False] * 3), (-1, -1))
    _cmp("match.first", M.match_send_slot(slots, 7, 11, 50, [False] * 3), (0, 50))
    _cmp("match.clip", M.match_send_slot(slots, 7, 11, 500, [False] * 3), (0, 100))
    _cmp("match.taken", M.match_send_slot(slots, 7, 11, 500, [True, False, False]),
         (1, 40))
    _cmp("match.miss", M.match_send_slot(slots, 7, 33, 50, [False] * 3), (-1, -1))
    _cmp("match.late", M.match_send_slot(slots, 7, 22, 10, [True, True, False]),
         (2, 10))
    _raises("match.empty", M.match_send_slot, [], 7, 11, 50, [])
    for bad in ({"size": -1}, {"addr": 0}, {"rkeys": [0, 6]}):
        s0 = dict(slots[0])
        s0.update(bad)
        _raises("match.bad%s" % sorted(bad)[0],
                M.match_send_slot, [s0] + slots[1:], 7, 11, 50, [False] * 3)
        checks += 1
    bad_n = [dict(slots[0], nreqs=9)] + slots[1:]
    _raises("match.nreqs9", M.match_send_slot, bad_n, 7, 11, 50, [False] * 9)
    checks += 8

    # last_wr_plan
    cases = [
        (1, 100, 5, BY_INDEX, False, 1 << 20, False, False),
        (1, 100, 1 << 21, BY_INDEX, True, 1 << 20, False, False),
        (1, 100, 1 << 21, BY_INDEX, True, 1 << 20, True, True),
        (1, 100, 1 << 21, BY_ID, True, 1 << 20, True, False),
        (4, 77, 4096, BY_ID, False, 1 << 20, False, False),
        (8, 1 << 33, 0, BY_ID, True, 0, False, False),
        (2, 0, 0, BY_INDEX, False, 0, False, False),
    ]
    for nreqs, rid, sz, sch, ar, thr, ro, lo in cases:
        got = M.last_wr_plan(nreqs, rid, sz, sch, ar, thr, ro, lo, 0xC000, 9,
                             [0x71, 0x72, 0x73])
        want = ref_last(nreqs, rid, sz, sch, ar, thr, ro, lo, 0xC000, 9,
                        [0x71, 0x72, 0x73])
        _cmp("last(%d,%d,%d,%d)" % (nreqs, rid, sz, sch), got, want)
        checks += 1
    _raises("last.nreqs0", M.last_wr_plan, 0, 1, 1, BY_ID, False, 0, False,
            False, 0, 0, [])
    _raises("last.scheme", M.last_wr_plan, 1, 1, 1, 7, False, 0, False, False,
            0, 0, [])
    checks += 2

    # plan_multisend / multisend_stats over randomized fabrics
    fabrics = [
        (1, [0], [0]),
        (2, [0, 1], [1, 0]),
        (4, [0, 1, 0, 1], [0, 0, 1, 1]),
        (8, TIMING_FABRIC["qp_dev"], TIMING_FABRIC["qp_remdev"]),
        (3, [0, 0, 1], [1, 1, 0]),
    ]
    rng = random.Random(99)
    for fi, (nqps, qd, qrd) in enumerate(fabrics):
        ndevs = max(qd) + 1
        rem = max(qrd) + 1
        for case in range(14):
            nreqs = 1 + (case % MAX_RECVS)
            sizes = [rng.choice(SIZE_MIX) for _ in range(nreqs)]
            if case == 0:
                sizes = [0] * nreqs
            if case == 1:
                sizes = [1] * nreqs
            data = [0x1000 + 0x40000 * r for r in range(nreqs)]
            raddrs = [0x900000 + 0x40000 * r for r in range(nreqs)]
            lkeys = [[rng.randrange(1, 1 << 20) for _ in range(ndevs)]
                     for _ in range(nreqs)]
            rkeys = [[rng.randrange(1, 1 << 20) for _ in range(rem)]
                     for _ in range(nreqs)]
            slot = (fi * 31 + case * 7) % FIFO_DEPTH
            rid = fi * 101 + case * 13
            got = M.plan_multisend(sizes, data, lkeys, raddrs, rkeys, slot,
                                    nqps, qd, qrd, rid)
            want = ref_plan(sizes, data, lkeys, raddrs, rkeys, slot, nqps, qd,
                            qrd, rid)
            _cmp("plan[f%d,c%d]" % (fi, case), got, want)
            _cmp("stats[f%d,c%d]" % (fi, case), M.multisend_stats(got),
                 ref_stats(want))
            # conservation: posted bytes never exceed the request size
            for r in range(nreqs):
                if got["posted"][r] != sizes[r]:
                    raise Fail("plan[f%d,c%d]: posted[%d]=%d != size %d"
                               % (fi, case, r, got["posted"][r], sizes[r]))
            checks += 2
    _raises("plan.nreqs0", M.plan_multisend, [], [], [], [], [], 0, 2, [0, 1],
            [0, 1], 0)
    _raises("plan.ragged", M.plan_multisend, [10, 20], [1], [[1]], [2],
            [[1]], 0, 2, [0, 1], [0, 1], 0)
    _raises("plan.negsize", M.plan_multisend, [-8], [1], [[1, 1]], [2],
            [[1, 1]], 0, 2, [0, 1], [0, 1], 0)
    checks += 3

    # drain_cts_fifo end to end
    for seed, nb in ((11, 40), (12, 37), (13, 23)):
        batches = make_batches(nb, 8, 2, 2, seed)
        got = M.drain_cts_fifo(batches, 8, TIMING_FABRIC["qp_dev"],
                               TIMING_FABRIC["qp_remdev"], BY_INDEX, True,
                               1 << 17, False, True,
                               TIMING_FABRIC["cmpls_addr"],
                               TIMING_FABRIC["cmpls_rkeys"])
        want = ref_drain(batches, 8, TIMING_FABRIC["qp_dev"],
                         TIMING_FABRIC["qp_remdev"], BY_INDEX, True, 1 << 17,
                         False, True, TIMING_FABRIC["cmpls_addr"],
                         TIMING_FABRIC["cmpls_rkeys"])
        if not want:
            raise Fail("drain seed %d produced an empty reference" % seed)
        _cmp("drain[seed%d]" % seed, got, want)
        checks += 1

    # an unmatched batch must be skipped entirely
    b = make_batches(3, 8, 2, 2, 77)
    b[1]["posts"] = b[1]["posts"][:-1] if len(b[1]["posts"]) > 1 else b[1]["posts"]
    got = M.drain_cts_fifo(b, 8, TIMING_FABRIC["qp_dev"],
                           TIMING_FABRIC["qp_remdev"], BY_ID, False, 0, True,
                           True, TIMING_FABRIC["cmpls_addr"],
                           TIMING_FABRIC["cmpls_rkeys"])
    want = ref_drain(b, 8, TIMING_FABRIC["qp_dev"], TIMING_FABRIC["qp_remdev"],
                     BY_ID, False, 0, True, True,
                     TIMING_FABRIC["cmpls_addr"], TIMING_FABRIC["cmpls_rkeys"])
    _cmp("drain.partial", got, want)
    checks += 1

    return checks


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def timing():
    import ib_multisend_planner as M
    batches = make_batches(TIMING_BATCHES, TIMING_NQPS, TIMING_NDEVS,
                           TIMING_REMDEVS, TIMING_SEED)
    args = (batches, TIMING_NQPS, TIMING_FABRIC["qp_dev"],
            TIMING_FABRIC["qp_remdev"], BY_INDEX, True, 1 << 17, False, True,
            TIMING_FABRIC["cmpls_addr"], TIMING_FABRIC["cmpls_rkeys"])
    # WARMUP on the identical full-size inputs (never a shrunken proxy).
    warm = M.drain_cts_fifo(*args)
    if not warm:
        raise Fail("timing warmup produced no plans")
    samples = []
    for _ in range(3):
        t0 = time.process_time()
        out = M.drain_cts_fifo(*args)
        t1 = time.process_time()
        samples.append((t1 - t0) * 1000.0)
        if len(out) != len(warm):
            raise Fail("timing run produced %d plans, warmup had %d"
                       % (len(out), len(warm)))
    return statistics.median(samples), len(warm), samples


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    res = {"mode": mode}
    if mode == "correctness":
        try:
            n = correctness()
            res.update({"correctness_ok": True, "checks": n})
        except Fail as e:
            res.update({"correctness_ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            res.update({"correctness_ok": False,
                        "error": "%s: %s" % (type(e).__name__, e)})
    elif mode == "timing":
        try:
            ms, nplans, samples = timing()
            res.update({"timing_ms": round(ms, 4), "plans": nplans,
                        "samples_ms": [round(s, 4) for s in samples],
                        "batches": TIMING_BATCHES})
        except Exception as e:  # noqa: BLE001
            res.update({"timing_ms": -1, "error": "%s: %s" % (type(e).__name__, e)})
    else:
        res.update({"error": "unknown mode"})
    print(TOKEN + " " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
