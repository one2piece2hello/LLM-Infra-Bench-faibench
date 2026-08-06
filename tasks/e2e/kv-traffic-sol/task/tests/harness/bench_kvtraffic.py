#!/usr/bin/env python3
"""FROZEN EVALUATION SURFACE — paged-KV TRAFFIC benchmark harness (reviewer-owned).

Uploaded fresh at scoring; never baked model-visible. ALL measurement happens HERE, from
outside the candidate:

  * every workload (page pool contents, block tables, per-request ranges, page-copy sets) is
    synthesised deterministically from a HIDDEN per-case seed inside this file;
  * the candidate is driven ONLY through the published entry contract (KVTrafficEngine);
  * timing is a harness-owned wall clock bracketed by FULL DEVICE syncs around a whole
    traffic step (so work pushed to a side stream cannot escape the window); the CUDA-event
    pair is recorded too and reported as a cross-check;
  * the minimum byte count is computed HERE from the case spec — the candidate never reports
    its own bytes, bandwidth or latency;
  * the pool footprint and the peak allocation are measured HERE with torch.cuda counters;
  * every timed block's OUTPUT is verified bit-exactly against the harness's own reference,
    with the destination poisoned first (a partial write cannot hide);
  * sol_fraction = (bytes_min / measured H20 peak HBM BW) / t_step -> physically <= 1;
    anything above PLAUSIBILITY_MAX_SOL is a hard fail.

Correctness runs FIRST and is a hard prerequisite: a failure scores 0 before any timing.
"""
import gc
import importlib.util
import json
import math
import os
import sys
import time

import torch

class HardFail(Exception):
    pass


class GateFail(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# HARNESS-OWNED PRIMITIVES, bound BEFORE any candidate module is imported.
# The candidate is loaded with importlib INTO THIS PROCESS (_load_module), so it shares `torch`
# and `time` with the measurement.  MEASURED 2026-07-27 (vacuity sweep V6): a candidate that was
# byte-for-byte the strong baseline plus `time.perf_counter = lambda: t0 + (real()-t0)/3.9`
# scored reward 0.718 (speedup 3.9036, all correctness cases green) for ZERO optimisation.
# Holding DIRECT references to the original function objects makes a later module-attribute or
# class-method patch unreachable from the measurement path, and _assert_pristine() turns the
# attempt itself into a HARD FAIL with a named reason.
_PERF = time.perf_counter
_SYNC = torch.cuda.synchronize
_EV_ELAPSED = torch.cuda.Event.elapsed_time
_EQUAL = torch.equal
_PRISTINE = (
    ("time.perf_counter", lambda: time.perf_counter, _PERF),
    ("torch.cuda.synchronize", lambda: torch.cuda.synchronize, _SYNC),
    ("torch.cuda.Event.elapsed_time", lambda: torch.cuda.Event.elapsed_time, _EV_ELAPSED),
    ("torch.equal", lambda: torch.equal, _EQUAL),
)


def _assert_pristine(tag):
    """reward.md 前置门 3 (作弊: modifying the bench harness).  Any replacement of a timing or
    comparison primitive the measurement depends on is a HARD FAIL with a named reason."""
    bad = []
    for name, getter, orig in _PRISTINE:
        try:
            if getter() is not orig:
                bad.append(name)
        except Exception:  # noqa: BLE001
            bad.append(name + "(unreadable)")
    if bad:
        raise HardFail(
            "harness_tamper_detected[%s]: the candidate replaced the harness measurement "
            "primitive(s) %s. Timing, bit-exactness and the plausibility bound are the "
            "evaluation surface; patching them is cheating (reward.md 前置门 3)."
            % (tag, ", ".join(bad)))


PEAK_HBM_GBPS = float(os.environ.get("KB_PEAK_HBM_GBPS", "3687.3"))
PLAUSIBILITY_MAX_SOL = 1.02
POISON = -12345.0
DEFAULT_WORKSPACE_MIB = 384
POOL_SLACK = 1.10


# ------------------------------------------------------------------ helpers
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HardFail("cannot load module from %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _elt(dt):
    return torch.tensor([], dtype=dt).element_size()


def _engine_cfg(cfg, device, max_pages_per_request):
    return {
        "num_layers": int(cfg["num_layers"]),
        "num_kv_heads": int(cfg["num_kv_heads"]),
        "head_size": int(cfg["head_size"]),
        "page_size": int(cfg["page_size"]),
        "dtype": cfg.get("dtype", "bfloat16"),
        "num_pages": int(cfg["num_pages"]),
        "max_batch": int(cfg["batch"]),
        "max_pages_per_request": int(max_pages_per_request),
        "device": device,
    }


def _nominal_pool_bytes(cfg):
    dt = _dtype(cfg.get("dtype", "bfloat16"))
    return (2 * int(cfg["num_layers"]) * int(cfg["num_pages"]) * int(cfg["page_size"])
            * int(cfg["num_kv_heads"]) * int(cfg["head_size"]) * _elt(dt))


def _row_bytes(cfg):
    dt = _dtype(cfg.get("dtype", "bfloat16"))
    return int(cfg["num_kv_heads"]) * int(cfg["head_size"]) * _elt(dt)


def _make_plan(bt, ctx, new, device):
    """The per-step plan handed to KVTrafficEngine.begin_step (both device + host mirrors)."""
    total = int(new.sum())
    return {
        "block_table": bt.to(device, non_blocking=True),
        "ctx_lens": ctx.to(device, non_blocking=True),
        "new_lens": new.to(device, non_blocking=True),
        "block_table_cpu": bt,
        "ctx_lens_cpu": ctx,
        "new_lens_cpu": new,
        "total_tokens": total,
        "batch": int(new.numel()),
    }, total


def _alloc_tables(cfg, g, seeds_used, share=False, shuffle=True):
    """Physical page assignment for every request (may be shuffled / shared)."""
    B, PAGE = int(cfg["batch"]), int(cfg["page_size"])
    npages = int(cfg["num_pages"])
    totals = seeds_used
    need = [(t + PAGE - 1) // PAGE for t in totals]
    mp = max(1, max(need))
    order = torch.randperm(npages, generator=g) if shuffle else torch.arange(npages)
    bt = torch.full((B, mp), -1, dtype=torch.int32)
    cursor = 0
    for b in range(B):
        if share and b > 0:
            # request b shares the first min(need[b], need[0]) pages of request 0
            nshare = min(need[b], need[0])
            bt[b, :nshare] = bt[0, :nshare]
            if need[b] > nshare:
                extra = need[b] - nshare
                bt[b, nshare:need[b]] = order[cursor:cursor + extra].to(torch.int32)
                cursor += extra
            continue
        bt[b, :need[b]] = order[cursor:cursor + need[b]].to(torch.int32)
        cursor += need[b]
    if cursor > npages:
        raise HardFail("case needs %d pages but num_pages=%d" % (cursor, npages))
    return bt, mp


def _content(cfg, g, totals, device):
    """Reference KV content per (layer, request): list[L][B] of [S_b, Hkv, D]."""
    L, B = int(cfg["num_layers"]), int(cfg["batch"])
    Hkv, D = int(cfg["num_kv_heads"]), int(cfg["head_size"])
    dt = _dtype(cfg.get("dtype", "bfloat16"))
    K, V = [], []
    for _ in range(L):
        kk, vv = [], []
        for b in range(B):
            s = max(1, totals[b])
            kk.append((torch.randn(s, Hkv, D, generator=g) * 0.5).to(dt).to(device))
            vv.append((torch.randn(s, Hkv, D, generator=g) * 0.5).to(dt).to(device))
        K.append(kk)
        V.append(vv)
    return K, V


def _apply_sharing(K, V, bt, totals, PAGE):
    """Requests whose leading pages are PHYSICALLY SHARED with request 0 must hold the same
    bytes there (real prefix sharing); otherwise two writers would disagree on one page."""
    B = len(totals)
    for b in range(1, B):
        j = 0
        while j < bt.shape[1] and int(bt[b, j]) >= 0 and int(bt[b, j]) == int(bt[0, j]):
            j += 1
        n = min(j * PAGE, totals[b], totals[0])
        if n <= 0:
            continue
        for l in range(len(K)):
            K[l][b][:n] = K[l][0][:n]
            V[l][b][:n] = V[l][0][:n]


def _pack(K, V, layer, ctx, new, device):
    """Pack the reference rows for one step (request-major) exactly as the contract requires."""
    B = ctx.numel()
    ks, vs = [], []
    for b in range(B):
        n = int(new[b])
        if n == 0:
            continue
        c = int(ctx[b])
        ks.append(K[layer][b][c:c + n])
        vs.append(V[layer][b][c:c + n])
    if not ks:
        Hkv, D = K[layer][0].shape[1], K[layer][0].shape[2]
        z = torch.empty(0, Hkv, D, dtype=K[layer][0].dtype, device=device)
        return z, z.clone()
    return torch.cat(ks, 0).contiguous(), torch.cat(vs, 0).contiguous()


def _poison(t):
    t.fill_(POISON)


def _footprint(tag, cid, base_alloc, nominal, extra=0):
    """Dual-measured pool footprint. Called after allocate() AND after the write phase, so a
    lazily-allocated shadow layout cannot slip past the first check."""
    got = torch.cuda.memory_allocated() - base_alloc
    budget = int(POOL_SLACK * nominal) + int(extra)
    if got > budget:
        raise GateFail("pool_footprint[%s/%s]: %d B live > %d B (%.2fx the nominal paged-pool "
                       "size%s)" % (cid, tag, got, budget, POOL_SLACK,
                                    " + %d MiB slack" % (extra >> 20) if extra else ""))
    return got


def _exact(got, exp, what, case_id):
    if tuple(got.shape) != tuple(exp.shape):
        raise GateFail("%s[%s]: shape %s != expected %s"
                       % (what, case_id, tuple(got.shape), tuple(exp.shape)))
    if got.dtype != exp.dtype:
        raise GateFail("%s[%s]: dtype %s != expected %s" % (what, case_id, got.dtype, exp.dtype))
    if not _EQUAL(got, exp):
        n = int((got != exp).sum())
        poisoned = int((got == torch.tensor(POISON, dtype=got.dtype)).sum())
        raise GateFail("%s[%s]: not bit-exact (%d of %d elements differ, %d still hold the "
                       "harness poison value = never written)"
                       % (what, case_id, n, got.numel(), poisoned))


# ------------------------------------------------------------------ correctness suite
def run_correctness_case(mod, cfg, device="cuda"):
    """One hidden correctness configuration. Bit-exact round-trip over the whole contract,
    plus the poison / source-mutation / stale-plan / footprint probes."""
    _assert_pristine("correctness case %s" % cfg.get("case_id"))
    cid = cfg["case_id"]
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
    B, PAGE, L = int(cfg["batch"]), int(cfg["page_size"]), int(cfg["num_layers"])
    Hkv, D = int(cfg["num_kv_heads"]), int(cfg["head_size"])
    dt = _dtype(cfg.get("dtype", "bfloat16"))
    totals = [int(x) for x in cfg["seq_lens"]]
    share = bool(cfg.get("share_pages"))
    bt, mp = _alloc_tables(cfg, g, totals, share=share, shuffle=bool(cfg.get("shuffle", True)))
    K, V = _content(cfg, g, totals, device)
    if share:
        _apply_sharing(K, V, bt, totals, PAGE)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    eng = mod.KVTrafficEngine(_engine_cfg(cfg, device, mp))
    eng.allocate()
    _SYNC()
    nominal = _nominal_pool_bytes(cfg)
    pool_alloc = _footprint("allocate", cid, base_alloc, nominal)

    # ---- write the whole context in arbitrary chunks (unaligned starts, partial tails) ----
    chunks = [int(x) for x in cfg.get("write_chunks", [PAGE + 1, PAGE, 3, 1, 7 * PAGE])]
    ci = 0
    pos = [0] * B
    while any(pos[b] < totals[b] for b in range(B)):
        step = chunks[ci % len(chunks)]
        ci += 1
        new = torch.tensor([min(step, max(0, totals[b] - pos[b])) for b in range(B)],
                           dtype=torch.int32)
        ctx = torch.tensor(pos, dtype=torch.int32)
        if int(new.sum()) == 0:
            break
        plan, total = _make_plan(bt, ctx, new, device)
        eng.begin_step(plan)
        for l in range(L):
            ks, vs = _pack(K, V, l, ctx, new, device)
            src_k, src_v = ks.clone(), vs.clone()
            eng.scatter(l, src_k, src_v)
            # source-mutation probe: the engine may not alias the caller's buffers
            src_k.fill_(POISON)
            src_v.fill_(POISON)
        for b in range(B):
            pos[b] += int(new[b])
    _SYNC()
    # second footprint measurement: a shadow layout built lazily on the write path is caught here
    # (the harness's own live tensors at this point are the reference content, allocated BEFORE
    # base_alloc, plus the small per-chunk packs which are freed each iteration).
    _footprint("after_write", cid, base_alloc, nominal,
               extra=int(cfg.get("workspace_mib", DEFAULT_WORKSPACE_MIB)) * (1 << 20))

    # ---- gather back, in a DIFFERENT chunking, bit-exact, into poisoned buffers ----
    checks = 0
    ranges = cfg.get("read_ranges") or [[[0] * B, list(totals)]]
    for (ctx_l, new_l) in ranges:
        ctx = torch.tensor([int(x) for x in ctx_l], dtype=torch.int32)
        new = torch.tensor([int(x) for x in new_l], dtype=torch.int32)
        for b in range(B):
            if int(ctx[b]) + int(new[b]) > totals[b]:
                raise HardFail("bad read range in case %s" % cid)
        plan, total = _make_plan(bt, ctx, new, device)
        eng.begin_step(plan)
        for l in range(L):
            ko = torch.empty(total, Hkv, D, dtype=dt, device=device)
            vo = torch.empty_like(ko)
            _poison(ko)
            _poison(vo)
            eng.gather(l, ko, vo)
            _SYNC()
            ek, ev = _pack(K, V, l, ctx, new, device)
            _exact(ko, ek, "gather_k", cid)
            _exact(vo, ev, "gather_v", cid)
            checks += 2
        del plan

    # ---- page-addressing probe: read request b through request (b+1)'s PAGES. Storage must be
    # addressed by the physical page the block table names, not by (request, position) — an
    # engine that keeps its own per-request arena and ignores the table fails here. ----
    if B >= 2:
        n = min(min(totals), 3 * PAGE + 1)
        if n > 0:
            rot = bt.clone()
            for b in range(B):
                rot[b] = bt[(b + 1) % B]
            ctx = torch.zeros(B, dtype=torch.int32)
            new = torch.full((B,), n, dtype=torch.int32)
            plan, total = _make_plan(rot, ctx, new, device)
            eng.begin_step(plan)
            for l in (0, L - 1):
                ko = torch.empty(total, Hkv, D, dtype=dt, device=device)
                vo = torch.empty_like(ko)
                _poison(ko)
                _poison(vo)
                eng.gather(l, ko, vo)
                _SYNC()
                ek = torch.cat([K[l][(b + 1) % B][:n] for b in range(B)]).contiguous()
                ev = torch.cat([V[l][(b + 1) % B][:n] for b in range(B)]).contiguous()
                _exact(ko, ek, "page_addressing_k", cid)
                _exact(vo, ev, "page_addressing_v", cid)
                checks += 2

    # ---- stale-plan probe: two plans back to back; the second must win ----
    if B >= 2 and min(totals) >= 4:
        ctxA = torch.tensor([0] * B, dtype=torch.int32)
        newA = torch.tensor([min(4, totals[b]) for b in range(B)], dtype=torch.int32)
        ctxB = torch.tensor([max(0, totals[b] - min(4, totals[b])) for b in range(B)],
                            dtype=torch.int32)
        newB = torch.tensor([min(4, totals[b]) for b in range(B)], dtype=torch.int32)
        pA, tA = _make_plan(bt, ctxA, newA, device)
        eng.begin_step(pA)
        pB, tB = _make_plan(bt, ctxB, newB, device)
        eng.begin_step(pB)
        ko = torch.empty(tB, Hkv, D, dtype=dt, device=device)
        vo = torch.empty_like(ko)
        _poison(ko)
        _poison(vo)
        eng.gather(0, ko, vo)
        _SYNC()
        ek, ev = _pack(K, V, 0, ctxB, newB, device)
        _exact(ko, ek, "stale_plan_k", cid)
        _exact(vo, ev, "stale_plan_v", cid)
        checks += 2

    # ---- copy_pages: duplicate whole pages, then read them through a redirected table ----
    ncp = int(cfg.get("n_copy_pages", 0))
    if ncp:
        used = set()
        for b in range(B):
            for j in range((totals[b] + PAGE - 1) // PAGE):
                v = int(bt[b, j])
                if v >= 0:
                    used.add(v)
        free = [p for p in range(int(cfg["num_pages"])) if p not in used]
        src_pages = sorted(used)[:ncp]
        if len(free) < len(src_pages):
            raise HardFail("case %s has no free pages for copy_pages" % cid)
        dst_pages = free[:len(src_pages)]
        sp = torch.tensor(src_pages, dtype=torch.int32, device=device)
        dp = torch.tensor(dst_pages, dtype=torch.int32, device=device)
        for l in range(L):
            eng.copy_pages(l, sp, dp)
        _SYNC()
        # request 0's first len(src_pages) pages are re-pointed at the copies
        remap = bt.clone()
        for b in range(B):
            for j in range((totals[b] + PAGE - 1) // PAGE):
                v = int(bt[b, j])
                if v in src_pages:
                    remap[b, j] = dst_pages[src_pages.index(v)]
        span = min(totals[0], len(src_pages) * PAGE)
        ctx = torch.tensor([0] * B, dtype=torch.int32)
        new = torch.tensor([min(span, totals[b]) if b == 0 else 0 for b in range(B)],
                           dtype=torch.int32)
        plan, total = _make_plan(remap, ctx, new, device)
        eng.begin_step(plan)
        for l in range(L):
            ko = torch.empty(total, Hkv, D, dtype=dt, device=device)
            vo = torch.empty_like(ko)
            _poison(ko)
            _poison(vo)
            eng.gather(l, ko, vo)
            _SYNC()
            ek, ev = _pack(K, V, l, ctx, new, device)
            _exact(ko, ek, "copy_pages_k", cid)
            _exact(vo, ev, "copy_pages_v", cid)
            checks += 2

    peak = torch.cuda.max_memory_allocated() - base_alloc
    allow = int(cfg.get("workspace_mib", DEFAULT_WORKSPACE_MIB)) * (1 << 20)
    ref_bytes = 2 * L * sum(max(1, t) for t in totals) * Hkv * D * _elt(dt)
    budget = int(POOL_SLACK * nominal) + allow + 4 * ref_bytes
    if peak > budget:
        raise GateFail("peak_memory[%s]: %d B > %d B (pool + %d MiB working allowance "
                       "+ the harness's own reference tensors)" % (cid, peak, budget,
                                                                  allow >> 20))
    res = {"case_id": cid, "checks": checks, "pool_bytes": int(pool_alloc),
           "pool_bytes_nominal": int(nominal), "peak_bytes": int(peak),
           "axes": cfg.get("axes", "")}
    try:
        eng.reset()
    except Exception as e:  # noqa: BLE001
        raise GateFail("reset() raised %s: %s" % (type(e).__name__, e))
    del eng, K, V
    gc.collect()
    torch.cuda.empty_cache()
    return res


def _mutate_content(K, V, ref, ctx, new, step):
    """Deterministically change the FIRST PACKED ROW of every layer between timed blocks, and
    mirror the same value into the harness's expected pack.

    Cost: 2 x L single-row writes, outside the timed window, so neither the timed interval nor
    `bytes_min` moves.  Effect: the pool write path and the expected bytes move TOGETHER, so an
    honest engine still matches bit-exactly, while a candidate that caches its gather output (or
    no-ops copy_pages and serves the verification from that cache) replays stale bytes and fails
    the per-block bit-exactness check.  MEASURED 2026-07-27 (vacuity sweep V6): before this, a
    table-blind memoising engine passed all 16 correctness cases AND every per-block verification
    and reported sol_fraction 7.714 — only the plausibility bound stopped it."""
    B = new.numel()
    b0 = 0
    while b0 < B and int(new[b0]) == 0:
        b0 += 1
    if b0 >= B:
        return
    pos = int(ctx[b0])
    for l in range(len(K)):
        kv = 0.125 * (1 + ((step * 7 + l * 3) % 5))
        K[l][b0][pos].fill_(kv)
        V[l][b0][pos].fill_(-kv)
        ref[l][0][0].copy_(K[l][b0][pos])
        ref[l][1][0].copy_(V[l][b0][pos])


# ------------------------------------------------------------------ timed suite
def _blocks_for(cfg, g, totals, device, nblocks):
    """One independent (block_table, ctx, new) plan per timed block (anti-cache: the work
    changes every block, and every block's output is verified)."""
    out = []
    for _ in range(nblocks):
        bt, mp = _alloc_tables(cfg, g, totals, share=bool(cfg.get("share_pages")),
                               shuffle=bool(cfg.get("shuffle", True)))
        out.append((bt, mp))
    return out


def run_timed_case(mod, cfg, device="cuda"):
    cid = cfg["case_id"]
    _assert_pristine("timed case %s (entry)" % cid)
    op = cfg["op"]
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
    B, PAGE, L = int(cfg["batch"]), int(cfg["page_size"]), int(cfg["num_layers"])
    Hkv, D = int(cfg["num_kv_heads"]), int(cfg["head_size"])
    dt = _dtype(cfg.get("dtype", "bfloat16"))
    ctx_l = [int(x) for x in cfg["ctx_lens"]]
    new_l = [int(x) for x in cfg["new_lens"]]
    totals = [ctx_l[b] + new_l[b] for b in range(B)]
    nblocks = int(cfg.get("timed_blocks", 10))
    warm = int(cfg.get("warmup_blocks", 3))
    ctx = torch.tensor(ctx_l, dtype=torch.int32)
    new = torch.tensor(new_l, dtype=torch.int32)
    T = int(new.sum())
    plans = _blocks_for(cfg, g, totals, device, nblocks + warm)
    mp = max(p[1] for p in plans)
    K, V = _content(cfg, g, totals, device)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    eng = mod.KVTrafficEngine(_engine_cfg(cfg, device, mp))
    eng.allocate()
    _SYNC()
    pool_alloc = torch.cuda.memory_allocated() - base_alloc
    nominal = _nominal_pool_bytes(cfg)
    if pool_alloc > int(POOL_SLACK * nominal):
        raise GateFail("pool_footprint[%s]: %d B > %d B (%.2fx nominal)"
                       % (cid, pool_alloc, int(POOL_SLACK * nominal), POOL_SLACK))

    row = _row_bytes(cfg)
    ncp = int(cfg.get("n_copy_pages", 0))
    if op == "copy_pages":
        bytes_min = 2 * 2 * L * ncp * PAGE * Hkv * D * _elt(dt)
    else:
        bytes_min = 2 * 2 * L * T * Hkv * D * _elt(dt)

    # harness-owned buffers (never handed to the engine to keep)
    kout = [torch.empty(T, Hkv, D, dtype=dt, device=device) for _ in range(L)]
    vout = [torch.empty(T, Hkv, D, dtype=dt, device=device) for _ in range(L)]
    ref = [_pack(K, V, l, ctx, new, device) for l in range(L)]

    def fill_pool(bt):
        """Write the whole context of every request through the engine (untimed setup)."""
        p, _t = _make_plan(bt, torch.zeros(B, dtype=torch.int32),
                           torch.tensor(totals, dtype=torch.int32), device)
        eng.begin_step(p)
        for l in range(L):
            ks, vs = _pack(K, V, l, torch.zeros(B, dtype=torch.int32),
                           torch.tensor(totals, dtype=torch.int32), device)
            eng.scatter(l, ks, vs)
        _SYNC()

    def verify_gather(bt):
        for l in range(L):
            _exact(kout[l], ref[l][0], "timed_gather_k", cid)
            _exact(vout[l], ref[l][1], "timed_gather_v", cid)

    def verify_scatter(bt):
        p, total = _make_plan(bt, ctx, new, device)
        eng.begin_step(p)
        for l in range(L):
            _poison(kout[l])
            _poison(vout[l])
            eng.gather(l, kout[l], vout[l])
        _SYNC()
        verify_gather(bt)

    copy_sets = []
    if op == "copy_pages":
        npg = int(cfg["num_pages"])
        for bi in range(nblocks + warm):
            perm = torch.randperm(npg, generator=g)
            copy_sets.append((perm[:ncp].to(torch.int32).to(device),
                              perm[ncp:2 * ncp].to(torch.int32).to(device),
                              perm[:ncp].tolist(), perm[ncp:2 * ncp].tolist()))

    times = []
    ev_times = []
    for bi in range(nblocks + warm):
        bt, _mp = plans[bi]
        # anti-memoisation: the reference content changes every block (see _mutate_content)
        _mutate_content(K, V, ref, ctx, new, bi)

        def _t_begin():
            """Harness-owned clock. A full DEVICE sync brackets the window, so work pushed to a
            side stream (or otherwise not joined) cannot escape the measurement; the CUDA-event
            pair is kept only as a cross-check."""
            _SYNC()
            a = torch.cuda.Event(True)
            t0 = _PERF()
            a.record()
            return a, t0

        def _t_end(a, t0):
            z = torch.cuda.Event(True)
            z.record()
            _SYNC()
            t1 = _PERF()
            return t1 - t0, _EV_ELAPSED(a, z) / 1e3

        if op == "gather":
            fill_pool(bt)
            plan, total = _make_plan(bt, ctx, new, device)
            for l in range(L):
                _poison(kout[l])
                _poison(vout[l])
            a, t0 = _t_begin()
            eng.begin_step(plan)
            for l in range(L):
                eng.gather(l, kout[l], vout[l])
            dt_wall, dt_ev = _t_end(a, t0)
            verify_gather(bt)
        elif op == "scatter":
            plan, total = _make_plan(bt, ctx, new, device)
            srcs = [(ref[l][0].clone(), ref[l][1].clone()) for l in range(L)]
            a, t0 = _t_begin()
            eng.begin_step(plan)
            for l in range(L):
                eng.scatter(l, srcs[l][0], srcs[l][1])
            dt_wall, dt_ev = _t_end(a, t0)
            for sk, sv in srcs:          # source-mutation probe on the TIMED path
                sk.fill_(POISON)
                sv.fill_(POISON)
            del srcs
            verify_scatter(bt)
        elif op == "copy_pages":
            sp, dp, sl, dl = copy_sets[bi]
            # seed the source pages with known content through the contract, then copy
            bt2 = torch.full((B, mp), -1, dtype=torch.int32)
            per_req = ncp // B
            for b in range(B):
                for j in range(per_req):
                    bt2[b, j] = sl[b * per_req + j]
            tot2 = [per_req * PAGE] * B
            p2, _t = _make_plan(bt2, torch.zeros(B, dtype=torch.int32),
                                torch.tensor(tot2, dtype=torch.int32), device)
            eng.begin_step(p2)
            for l in range(L):
                ks, vs = _pack(K, V, l, torch.zeros(B, dtype=torch.int32),
                               torch.tensor(tot2, dtype=torch.int32), device)
                eng.scatter(l, ks, vs)
            _SYNC()
            a, t0 = _t_begin()
            for l in range(L):
                eng.copy_pages(l, sp, dp)
            dt_wall, dt_ev = _t_end(a, t0)
            # verify: read request b's span through the DESTINATION pages
            bt3 = torch.full((B, mp), -1, dtype=torch.int32)
            for b in range(B):
                for j in range(per_req):
                    bt3[b, j] = dl[b * per_req + j]
            ctx3 = torch.zeros(B, dtype=torch.int32)
            new3 = torch.tensor(tot2, dtype=torch.int32)
            p3, t3 = _make_plan(bt3, ctx3, new3, device)
            eng.begin_step(p3)
            ek, ev = None, None
            for l in range(L):
                ko = torch.empty(t3, Hkv, D, dtype=dt, device=device)
                vo = torch.empty_like(ko)
                _poison(ko)
                _poison(vo)
                eng.gather(l, ko, vo)
                _SYNC()
                ek, ev = _pack(K, V, l, ctx3, new3, device)
                _exact(ko, ek, "timed_copy_k", cid)
                _exact(vo, ev, "timed_copy_v", cid)
                del ko, vo
        else:
            raise HardFail("unknown op %r in case %s" % (op, cid))
        if bi >= warm:
            times.append(dt_wall)
            ev_times.append(dt_ev)

    _assert_pristine("timed case %s (after timing)" % cid)
    peak = torch.cuda.max_memory_allocated() - base_alloc
    allow = int(cfg.get("workspace_mib", DEFAULT_WORKSPACE_MIB)) * (1 << 20)
    harness_bytes = 4 * 2 * L * T * Hkv * D * _elt(dt) + 2 * L * sum(totals) * Hkv * D * _elt(dt)
    budget = int(POOL_SLACK * nominal) + allow + harness_bytes
    if peak > budget:
        raise GateFail("peak_memory[%s]: %d B > %d B (pool + %d MiB working allowance + the "
                       "harness's own buffers)" % (cid, peak, budget, allow >> 20))

    ts = sorted(times)
    t_med = ts[len(ts) // 2]
    evs = sorted(ev_times)
    sol_s = bytes_min / (PEAK_HBM_GBPS * 1e9)
    sol = sol_s / t_med
    if sol > PLAUSIBILITY_MAX_SOL:
        raise HardFail("implausible sol_fraction %.3f (> %.2f) on case %s: the minimum bytes "
                       "cannot move faster than the measured HBM peak"
                       % (sol, PLAUSIBILITY_MAX_SOL, cid))
    res = {"case_id": cid, "op": op, "step_time_median_s": t_med, "step_time_min_s": ts[0],
           "step_time_p90_s": ts[min(len(ts) - 1, int(0.9 * (len(ts) - 1)))],
           "step_time_spread_pct": 100.0 * (ts[-1] - ts[0]) / max(ts[0], 1e-9),
           "cuda_event_time_median_s": evs[len(evs) // 2] if evs else None,
           "bytes_min": int(bytes_min), "row_bytes": int(row),
           "achieved_gbps": bytes_min / t_med / 1e9,
           "sol_time_s": sol_s, "sol_fraction": sol,
           "pool_bytes": int(pool_alloc), "pool_bytes_nominal": int(nominal),
           "peak_bytes": int(peak), "n_timed_blocks": len(times),
           "tokens": T, "layers": L,
           "content_mutations_between_blocks": nblocks + warm}
    try:
        eng.reset()
    except Exception:  # noqa: BLE001
        pass
    del eng, K, V, kout, vout, ref
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ driver
def run_suite(mod_path, suite, device="cuda", timed_only=False):
    mod = _load_module(mod_path, "kb_kv_impl")
    if not hasattr(mod, "KVTrafficEngine"):
        raise HardFail("entry contract: class KVTrafficEngine not found in %s" % mod_path)
    # `timed_only` is used by the ABBA repeat measurements: the correctness suite is a hard
    # prerequisite that already ran (and passed) in the FULL pass, so re-running it on every
    # repeat only burns wall clock.  The per-case bit-exactness / poison / alias probes that
    # live INSIDE run_timed_case still run on every repeat, so a repeat can never be a
    # correctness-free fast path.
    corr = ([] if timed_only
            else [run_correctness_case(mod, c, device) for c in suite.get("correctness_cases", [])])
    timed = [run_timed_case(mod, c, device) for c in suite["timed_cases"]]
    geo = math.exp(sum(math.log(max(r["sol_fraction"], 1e-12)) for r in timed) / len(timed))
    return corr, timed, geo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timed-only", action="store_true",
                    help="skip the standalone correctness suite (ABBA repeat measurements only; "
                         "the in-case bit-exactness/poison/alias probes still run)")
    a = ap.parse_args()
    suite = json.load(open(a.suite))
    payload = {"suite": suite.get("name"), "impl": a.impl, "ts": time.time(),
               "peak_hbm_gbps": PEAK_HBM_GBPS,
               "timed_only": bool(a.timed_only),
               "expected_case_count": len(suite["timed_cases"]),
               "expected_correctness_case_count": (
                   0 if a.timed_only else len(suite.get("correctness_cases", [])))}
    try:
        corr, timed, geo = run_suite(a.impl, suite, timed_only=a.timed_only)
        payload.update({"status": "ok", "correctness_cases": corr, "cases": timed,
                        "geomean_sol_fraction": geo, "observed_case_count": len(timed),
                        "observed_correctness_case_count": len(corr)})
    except GateFail as e:
        payload.update({"status": "gate_fail", "reason": str(e)})
    except HardFail as e:
        payload.update({"status": "hard_fail", "reason": str(e)})
    except Exception as e:  # noqa: BLE001
        payload.update({"status": "hard_fail", "reason": "%s: %s" % (type(e).__name__, e)})
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("cases", "correctness_cases")}, indent=1))
    if payload.get("status") == "ok":
        for r in payload["cases"]:
            print("  %-30s %-11s sol=%.4f  %8.1f GB/s  %7.3f ms  spread=%.0f%%"
                  % (r["case_id"], r["op"], r["sol_fraction"], r["achieved_gbps"],
                     r["step_time_median_s"] * 1e3, r["step_time_spread_pct"]))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
