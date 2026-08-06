#!/usr/bin/env python3
"""FROZEN EVALUATION SURFACE — varlen/causal PREFILL attention benchmark harness (reviewer-owned).

Uploaded fresh at scoring; never baked model-visible. ALL measurement happens HERE, from outside
the candidate:
  * deterministic workload synthesis from a hidden spec (per-case seed / ragged length mix)
  * the candidate is driven ONLY through the published entry contract
  * per-call latency from CUDA events recorded by THIS process, median over blocks
  * the fp32 causal reference is recomputed HERE and gates the run (parity)
  * anti-cheat: poison-sentinel write probe, stale/memoisation probe, input-immutability check,
    query rotation across timed blocks, workspace-allocation budget
  * sol_fraction = (minimum causal attention FLOPs / MEASURED H20 dense bf16 peak) / call time
    -> a kernel cannot exceed the tensor-core peak; > PLAUSIBILITY_MAX_SOL is a hard fail
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
# The candidate is loaded with importlib INTO THIS PROCESS (run_case/_load_module), so it shares
# `torch` and `time` with the measurement.  MEASURED 2026-07-27 (vacuity sweep V6): a candidate
# that is byte-for-byte the strong baseline plus `torch.cuda.Event.elapsed_time = lambda ...: t/f`
# reached the timed phase and reported an f-times-faster kernel.  Holding DIRECT references to the
# original function objects here makes a later module-attribute or class-method patch unreachable
# from the measurement path, and _assert_pristine() turns the attempt itself into a HARD FAIL.
_EV_ELAPSED = torch.cuda.Event.elapsed_time
_SYNC = torch.cuda.synchronize
_EQUAL = torch.equal
_SOFTMAX = torch.softmax
_MATMUL = torch.matmul
_EINSUM = torch.einsum
_PERF = time.perf_counter
_PRISTINE = (
    ("torch.cuda.Event.elapsed_time", lambda: torch.cuda.Event.elapsed_time, _EV_ELAPSED),
    ("torch.cuda.synchronize", lambda: torch.cuda.synchronize, _SYNC),
    ("torch.equal", lambda: torch.equal, _EQUAL),
    ("torch.softmax", lambda: torch.softmax, _SOFTMAX),
    ("torch.matmul", lambda: torch.matmul, _MATMUL),
    ("torch.einsum", lambda: torch.einsum, _EINSUM),
    ("time.perf_counter", lambda: time.perf_counter, _PERF),
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
            "primitive(s) %s. Timing, parity and the plausibility bound are the evaluation "
            "surface; patching them is cheating (reward.md 前置门 3)." % (tag, ", ".join(bad)))


PEAK_TFLOPS = float(os.environ.get("KB_PEAK_BF16_TFLOPS", "137.94"))
PLAUSIBILITY_MAX_SOL = 1.15
WORKSPACE_ALLOWANCE_BYTES = 3 * (1 << 30)
POISON = 1024.0        # exactly representable in bf16 AND fp16, and unreachable by a real output


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


def _entry(mod):
    if not hasattr(mod, "VarlenPrefillAttention"):
        raise HardFail("entry contract: class VarlenPrefillAttention not found")
    return getattr(mod, "VarlenPrefillAttention")


# --------------------------------------------------------------------------- workloads
def _pack(seq_lens, Hq, Hkv, D, dt, device, seed):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    tot = int(sum(seq_lens))
    q = (torch.randn(max(tot, 1), Hq, D, generator=g) * 0.5).to(dt).to(device)[:tot]
    k = (torch.randn(max(tot, 1), Hkv, D, generator=g) * 0.5).to(dt).to(device)[:tot]
    v = (torch.randn(max(tot, 1), Hkv, D, generator=g) * 0.5).to(dt).to(device)[:tot]
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor([int(s) for s in seq_lens], dtype=torch.int32), 0)
    return q.contiguous(), k.contiguous(), v.contiguous(), cu.to(device)


def min_causal_flops(seq_lens, Hq, D):
    """The FLOPs any correct causal attention must do: 2 per MAC for QK^T + 2 for PV,
    over the lower-triangular pair count of each sequence."""
    return sum(4.0 * Hq * D * (int(s) * (int(s) + 1) / 2.0) for s in seq_lens)


def _cu_list(seq_lens):
    out = [0]
    for s in seq_lens:
        out.append(out[-1] + int(s))
    return out


# --------------------------------------------------------------------------- reference
def _ref_rows(q, k, v, cul, seq_idx, rows, Hq, Hkv, D):
    """fp32 causal reference for a list of (sequence, row) pairs. Returns [n, Hq, D]."""
    rep = Hq // Hkv
    scale = 1.0 / math.sqrt(D)
    outs = []
    for (i, r) in zip(seq_idx, rows):
        a = cul[i]
        qq = q[a + r].float()
        kk = k[a:a + r + 1].float()
        vv = v[a:a + r + 1].float()
        if rep > 1:
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
        logits = _EINSUM("hd,shd->hs", qq, kk) * scale
        p = _SOFTMAX(logits, dim=-1)
        outs.append(_EINSUM("hs,shd->hd", p, vv))
    return torch.stack(outs) if outs else torch.zeros(0, Hq, D, device=q.device)


def _ref_full_seq(q, k, v, a, n, Hq, Hkv, D, chunk=256):
    """fp32 causal reference for EVERY row of one sequence (used by the edge suite)."""
    rep = Hq // Hkv
    scale = 1.0 / math.sqrt(D)
    kk = k[a:a + n].float()
    vv = v[a:a + n].float()
    if rep > 1:
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
    kk = kk.transpose(0, 1)                      # [Hq, n, D]
    vv = vv.transpose(0, 1)
    out = torch.empty(n, Hq, D, device=q.device, dtype=torch.float32)
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        qq = q[a + c0:a + c1].float().transpose(0, 1)          # [Hq, c, D]
        s = _MATMUL(qq, kk.transpose(1, 2)) * scale       # [Hq, c, n]
        rows = torch.arange(c0, c1, device=q.device).unsqueeze(1)
        cols = torch.arange(n, device=q.device).unsqueeze(0)
        s = s.masked_fill((cols > rows).unsqueeze(0), float("-inf"))
        p = _SOFTMAX(s, dim=-1)
        out[c0:c1] = _MATMUL(p, vv).transpose(0, 1)
    return out


def _rel_err(got, ref):
    """Scale-aware PER-ROW error: for each query row, the max elementwise deviation relative to
    the RMS of that row's fp32 reference; the reported figure is the worst row.
    (A plain per-element relative error explodes on the near-zero elements that long-context
    attention outputs are full of, and would measure float noise instead of correctness.)"""
    if got.numel() == 0:
        return 0.0
    g = got.float().reshape(got.shape[0], -1)
    r = ref.float().reshape(ref.shape[0], -1)
    rms = r.pow(2).mean(dim=1).sqrt()
    return ((g - r).abs().amax(dim=1) / (rms + 1e-3)).max().item()


# --------------------------------------------------------------------------- checks
def _parity_rows(cfg, seq_lens, Hq, Hkv, D):
    """Deterministic, hidden set of (sequence, row) pairs: the first/second/middle/last row of
    every sequence plus seeded random rows, capped."""
    cap = int(cfg.get("parity_rows", 96))
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) ^ 0xA11CE)
    pairs = []
    for i, s in enumerate(seq_lens):
        s = int(s)
        if s <= 0:
            continue
        for r in {0, min(1, s - 1), s // 2, s - 1}:
            pairs.append((i, int(r)))
    nz = [i for i, s in enumerate(seq_lens) if int(s) > 0]
    while len(pairs) < cap and nz:
        i = nz[int(torch.randint(0, len(nz), (1,), generator=g).item())]
        r = int(torch.randint(0, int(seq_lens[i]), (1,), generator=g).item())
        pairs.append((i, r))
    pairs = sorted(set(pairs))[:cap]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _check_out(out, ret, tot, Hq, D, dt, tag):
    if ret is None:
        raise GateFail("%s: forward() returned None; it must return the output tensor" % tag)
    if ret.data_ptr() != out.data_ptr():
        raise GateFail("%s: forward() must write into the provided `out` tensor and return it "
                       "(returned a different buffer)" % tag)
    if tuple(ret.shape) != (tot, Hq, D):
        raise GateFail("%s: forward() output shape %s, expected %s"
                       % (tag, tuple(ret.shape), (tot, Hq, D)))
    if ret.dtype != dt:
        raise GateFail("%s: forward() output dtype %s, expected %s" % (tag, ret.dtype, dt))
    if tot and not torch.isfinite(ret.float()).all():
        raise GateFail("%s: forward() produced non-finite values" % tag)
    if tot:
        sentinel = torch.tensor(POISON, device=ret.device, dtype=dt)
        if bool((ret == sentinel).any().item()):
            raise GateFail("%s: forward() left part of the output buffer unwritten "
                           "(the pre-set sentinel value survived)" % tag)


def _mutate_workload(sets, step):
    """Deterministically change ONE value-tensor row of every rotation between timed blocks.

    Cost: one Hkv x D row write per rotation, OUTSIDE the CUDA-event window, so neither the timed
    interval nor the FLOP count moves.  Effect: every query row of sequence 0 attends token 0, so
    every one of that sequence's outputs changes and no cached result stays valid."""
    for j, (q, k, v, cu) in enumerate(sets):
        if v.shape[0] == 0:
            continue
        v[0].fill_(0.125 * (1 + ((step * 7 + j * 3) % 5)))


# --------------------------------------------------------------------------- edge suite
def run_edges(mod_path, edges, device="cuda"):
    """FULL fp32 parity over tiny degenerate varlen batches. Untimed. Fail => reward 0."""
    mod = _load_module(mod_path, "kb_edge_impl")
    _assert_pristine("edge suite after import")
    cls = _entry(mod)
    worst = 0.0
    n_checked = 0
    for e in edges["cases"]:
        sl = [int(x) for x in e["seq_lens"]]
        Hq, Hkv, D = int(e["num_q_heads"]), int(e["num_kv_heads"]), int(e["head_size"])
        dt = _dtype(e.get("dtype", "bfloat16"))
        tol = float(e.get("parity_tol", edges.get("parity_tol", 0.10)))
        q, k, v, cu = _pack(sl, Hq, Hkv, D, dt, device, e["seed"])
        tot = int(sum(sl))
        cfg = {"num_q_heads": Hq, "num_kv_heads": Hkv, "head_size": D,
               "dtype": e.get("dtype", "bfloat16"), "device": device,
               "max_num_seqs": len(sl), "max_seq_len": max([1] + sl),
               "max_total_tokens": max(tot, 1), "causal": True,
               "softmax_scale": 1.0 / math.sqrt(D)}
        impl = cls(cfg)
        impl.prepare()
        out = torch.full((tot, Hq, D), POISON, device=device, dtype=dt)
        keep = (q.clone(), k.clone(), v.clone())
        ret = impl.forward(q, k, v, cu, max([0] + sl), out)
        _check_out(out, ret, tot, Hq, D, dt, "edge[%s]" % e["case_id"])
        for nm, cur, orig in (("q", q, keep[0]), ("k", k, keep[1]), ("v", v, keep[2])):
            if not _EQUAL(cur, orig):
                raise GateFail("edge[%s]: forward() mutated its input tensor %s"
                               % (e["case_id"], nm))
        cul = _cu_list(sl)
        for i, s in enumerate(sl):
            if int(s) == 0:
                continue
            ref = _ref_full_seq(q, k, v, cul[i], int(s), Hq, Hkv, D)
            err = _rel_err(ret[cul[i]:cul[i] + int(s)], ref)
            n_checked += int(s)
            worst = max(worst, err)
            if err > tol:
                raise GateFail("edge[%s] seq %d (len %d): parity %.5g > tol %.5g — every query "
                               "row must attend its own causal prefix exactly"
                               % (e["case_id"], i, int(s), err, tol))
        del impl, q, k, v, out
        gc.collect()
        torch.cuda.empty_cache()
    return {"edge_cases": len(edges["cases"]), "edge_rows_checked": n_checked,
            "edge_worst_parity": worst}


# --------------------------------------------------------------------------- timed case
def run_case(mod_path, cfg, device="cuda"):
    mod = _load_module(mod_path, "kb_impl_" + str(abs(hash(cfg["case_id"]))))
    _assert_pristine("case[%s] after import" % cfg["case_id"])
    cls = _entry(mod)

    sl = [int(x) for x in cfg["seq_lens"]]
    Hq, Hkv, D = int(cfg["num_q_heads"]), int(cfg["num_kv_heads"]), int(cfg["head_size"])
    dt = _dtype(cfg.get("dtype", "bfloat16"))
    tot = int(sum(sl))
    mx = max(sl)
    cul = _cu_list(sl)
    n_rot = int(cfg.get("query_rotations", 3))
    tol = float(cfg.get("parity_tol", 0.10))

    gc.collect()
    torch.cuda.empty_cache()

    # --- harness-owned inputs: n_rot independent workloads (kills memoisation across blocks) ---
    sets = [_pack(sl, Hq, Hkv, D, dt, device, int(cfg["seed"]) + 7919 * j) for j in range(n_rot)]
    outs = [torch.full((tot, Hq, D), POISON, device=device, dtype=dt) for _ in range(n_rot)]
    clones = [(a.clone(), b.clone(), c.clone()) for (a, b, c, _) in sets]
    _SYNC()

    build_cfg = {"num_q_heads": Hq, "num_kv_heads": Hkv, "head_size": D,
                 "dtype": cfg.get("dtype", "bfloat16"), "device": device,
                 "max_num_seqs": len(sl), "max_seq_len": mx, "max_total_tokens": tot,
                 "causal": True, "softmax_scale": 1.0 / math.sqrt(D)}

    w0 = torch.cuda.memory_allocated()
    impl = cls(build_cfg)
    impl.prepare()
    _SYNC()
    workspace_bytes = max(0, torch.cuda.memory_allocated() - w0)

    # ---------------- correctness gate (recomputed HERE, in fp32) ----------------
    si, ri = _parity_rows(cfg, sl, Hq, Hkv, D)
    parity = {"max_err": 0.0, "rows": 0}
    first_out = None
    for j in range(n_rot):
        q, k, v, cu = sets[j]
        out = outs[j]
        out.fill_(POISON)
        ret = impl.forward(q, k, v, cu, mx, out)
        _check_out(out, ret, tot, Hq, D, dt, "case[%s] set%d" % (cfg["case_id"], j))
        # input immutability: the candidate may not scribble on the harness's tensors
        for nm, cur, orig in (("q", q, clones[j][0]), ("k", k, clones[j][1]),
                              ("v", v, clones[j][2])):
            if not _EQUAL(cur, orig):
                raise GateFail("case[%s]: forward() mutated its input tensor %s"
                               % (cfg["case_id"], nm))
        idx = torch.tensor([cul[i] + r for i, r in zip(si, ri)], device=device,
                           dtype=torch.long)
        got = ret.index_select(0, idx)
        ref = _ref_rows(q, k, v, cul, si, ri, Hq, Hkv, D)
        err = _rel_err(got, ref)
        parity["max_err"] = max(parity["max_err"], err)
        parity["rows"] += len(si)
        if err > tol:
            raise GateFail("parity[%s] set%d: %.5g > tol %.5g (the output must match the fp32 "
                           "causal reference over each query's FULL prefix)"
                           % (cfg["case_id"], j, err, tol))
        # on selected cases, check EVERY row (not just the sampled ones)
        if j == 0 and cfg.get("full_parity"):
            for i, s in enumerate(sl):
                s = int(s)
                if s == 0:
                    continue
                fref = _ref_full_seq(q, k, v, cul[i], s, Hq, Hkv, D)
                ferr = _rel_err(ret[cul[i]:cul[i] + s], fref)
                parity["max_err"] = max(parity["max_err"], ferr)
                parity["rows"] += s
                del fref
                if ferr > tol:
                    raise GateFail("full_parity[%s] seq %d (len %d): %.5g > tol %.5g — EVERY "
                                   "query row must attend its own causal prefix"
                                   % (cfg["case_id"], i, s, ferr, tol))
            parity["full_checked"] = True
        if j == 0:
            first_out = ret.clone()
        elif first_out is not None and _EQUAL(ret, first_out):
            raise GateFail("case[%s]: two DIFFERENT input workloads produced a bit-identical "
                           "output — the result is not being computed from the inputs"
                           % cfg["case_id"])
    del first_out

    # ---------------- timed phase ----------------
    block = int(cfg.get("timed_block", 4))
    n_blocks = int(cfg.get("timed_blocks", 10))
    warmup = int(cfg.get("warmup_calls", 5))
    for w in range(warmup):
        q, k, v, cu = sets[w % n_rot]
        impl.forward(q, k, v, cu, mx, outs[w % n_rot])
    _SYNC()

    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    times = []
    call = 0
    # 🔴 ANTI-MEMOISATION (MEASURED 2026-07-27, vacuity sweep V6).  The correctness phase above is
    #    the ONLY place the output used to be checked, and it reuses the SAME n_rot tensors for
    #    warmup and for every timed block — so a candidate whose forward() caches its result under
    #    the identity (or the contents) of its inputs passed every probe here (3 different
    #    workloads DID give 3 different correct outputs) and then replayed a copy for the whole
    #    timed phase.  Measured: sol_fraction 9.122 of the dense peak, i.e. only the plausibility
    #    bound stood between a zero-work replay and a full-marks reward.
    #    Fix: the workload is MUTATED between timed blocks (one value-tensor row per rotation,
    #    outside the CUDA-event window, so timing and FLOPs are unchanged — every query row of
    #    sequence 0 attends token 0, so every one of its outputs must change), and after the timed
    #    phase the output the LAST timed call left behind is re-checked against the fp32 reference
    #    for the CURRENT inputs.  A replayed cache is stale and fails; an honest kernel passes.
    last_block = {}
    for b_i in range(n_blocks):
        _mutate_workload(sets, b_i)
        _SYNC()
        a = torch.cuda.Event(True)
        z = torch.cuda.Event(True)
        a.record()
        for _ in range(block):
            j = call % n_rot
            impl.forward(sets[j][0], sets[j][1], sets[j][2], sets[j][3], mx, outs[j])
            last_block[j] = b_i
            call += 1
        z.record()
        _SYNC()
        times.append(_EV_ELAPSED(a, z) / 1e3 / block)
    peak_extra = max(0, torch.cuda.max_memory_allocated() - base_alloc)

    # replay check: only the rotations whose LAST timed write happened in the FINAL block can be
    # compared against the final mutated inputs.
    replay_checked = 0
    for j in range(n_rot):
        if last_block.get(j) != n_blocks - 1 or tot == 0:
            continue
        q, k, v, cu = sets[j]
        idx = torch.tensor([cul[i] + r for i, r in zip(si, ri)], device=device, dtype=torch.long)
        got = outs[j].index_select(0, idx)
        ref = _ref_rows(q, k, v, cul, si, ri, Hq, Hkv, D)
        err = _rel_err(got, ref)
        parity["max_err"] = max(parity["max_err"], err)
        replay_checked += len(si)
        if err > tol:
            raise GateFail(
                "timed_replay[%s] rot%d: %.5g > tol %.5g — the output left by the LAST timed call "
                "does not match the fp32 causal reference for the CURRENT inputs. The workload is "
                "mutated between timed blocks, so a cached/memoised/replayed result is stale and "
                "the timed phase did not actually compute the attention."
                % (cfg["case_id"], j, err, tol))
    _assert_pristine("case[%s] after timing" % cfg["case_id"])

    budget = WORKSPACE_ALLOWANCE_BYTES
    if workspace_bytes + peak_extra > budget:
        raise GateFail("workspace_budget[%s]: %d B persistent + %d B transient > %d B allowed — "
                       "the implementation must stream the attention computation, not materialise "
                       "the full score matrix" % (cfg["case_id"], workspace_bytes, peak_extra,
                                                  budget))

    mf = min_causal_flops(sl, Hq, D)
    sol_s = mf / (PEAK_TFLOPS * 1e12)
    ts = sorted(times)
    t_med = ts[len(ts) // 2]
    sol_fraction = sol_s / t_med
    if sol_fraction > PLAUSIBILITY_MAX_SOL:
        raise HardFail("implausible sol_fraction %.3f (> %.2f) on case %s: the required matmul "
                       "FLOPs cannot be executed faster than the measured dense tensor-core peak"
                       % (sol_fraction, PLAUSIBILITY_MAX_SOL, cfg["case_id"]))

    res = {
        "case_id": cfg["case_id"],
        "n_seq": len(sl), "total_tokens": tot, "max_seqlen": mx,
        "num_q_heads": Hq, "num_kv_heads": Hkv, "head_size": D,
        "min_causal_flops": mf,
        "sol_time_s": sol_s,
        "call_time_median_s": t_med,
        "call_time_min_s": ts[0],
        "call_time_p90_s": ts[min(len(ts) - 1, int(0.9 * len(ts)))],
        "call_time_spread_pct": 100.0 * (ts[-1] - ts[0]) / max(ts[0], 1e-12),
        "achieved_tflops": mf / t_med / 1e12,
        "sol_fraction": sol_fraction,
        "parity_max_err": parity["max_err"],
        "parity_rows": parity["rows"],
        "parity_full_checked": bool(parity.get("full_checked", False)),
        "timed_replay_rows_checked": replay_checked,
        "workspace_bytes": int(workspace_bytes),
        "peak_transient_bytes": int(peak_extra),
        "workspace_budget_bytes": int(budget),
        "n_timed_blocks": len(times), "timed_block_size": block,
    }
    del impl, sets, outs, clones
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_suite(mod_path, suite, device="cuda", timed_only=False):
    # `timed_only` is used by the ABBA repeat measurements: the degenerate-shape EDGE suite is a
    # hard prerequisite that already ran (and passed) in the FULL pass, so re-running it on every
    # repeat only burns wall clock. The per-case fp32 causal parity / sentinel / stale-output /
    # input-immutability / workspace-budget probes live INSIDE run_case and still run on EVERY
    # repeat, so a repeat can never be a correctness-free fast path.
    edge = {}
    if suite.get("edges") and not timed_only:
        edge = run_edges(mod_path, suite["edges"], device=device)
    results = [run_case(mod_path, cfg, device=device) for cfg in suite["cases"]]
    geo = math.exp(sum(math.log(max(r["sol_fraction"], 1e-12)) for r in results) / len(results))
    return results, geo, edge


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timed-only", action="store_true",
                    help="skip the standalone degenerate-shape EDGE suite (ABBA repeat "
                         "measurements only; the per-case parity/sentinel/stale/immutability/"
                         "workspace probes still run)")
    a = ap.parse_args()
    suite = json.load(open(a.suite))
    payload = {"suite": suite.get("name"), "impl": a.impl, "ts": time.time(),
               "peak_bf16_tflops": PEAK_TFLOPS,
               "timed_only": bool(a.timed_only),
               "expected_case_count": len(suite["cases"])}
    try:
        res, geo, edge = run_suite(a.impl, suite, timed_only=a.timed_only)
        payload.update({"status": "ok", "cases": res, "geomean_sol_fraction": geo,
                        "observed_case_count": len(res)})
        payload.update(edge)
    except GateFail as e:
        payload.update({"status": "gate_fail", "reason": str(e)})
    except HardFail as e:
        payload.update({"status": "hard_fail", "reason": str(e)})
    except Exception as e:  # noqa: BLE001
        payload.update({"status": "hard_fail", "reason": "%s: %s" % (type(e).__name__, e)})
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, indent=1))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
