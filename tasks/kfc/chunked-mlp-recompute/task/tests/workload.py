#!/usr/bin/env python3
"""Verifier workload for chunked-mlp-recompute. GPU task (anchor calibrated on H20).

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden bf16 gated-MLP
forward+backward workloads. Correctness = match a SEEDED fp32 mathematical reference within
rtol=atol=2e-2 on BOTH returned tensors (y and dx). The reference IS the disclosed contract math
(forward + manual input-gradient backward of a SwiGLU MLP block), computed independently here —
NOT the fast/low-memory oracle, which is not baked into the image.

perf_metric = peak_bytes (LOWER is better). The timing mode measures the peak GPU allocator
high-water `torch.cuda.max_memory_allocated()` around a `custom_kernel` call and reports it in the
`timing_ms` field, so the shared, metric-agnostic test.sh computes
    vs_oracle = oracle_ms / candidate_ms = oracle_peak_bytes / candidate_peak_bytes
(a candidate whose peak is smaller than the oracle's -> vs_oracle > 1; equal -> 1.0; the
materialize-everything baseline2, which holds all large [T,I] intermediates resident for the
backward -> 0 < vs_oracle < 1). The peak is measured HERE, so it cannot be faked; a lower peak
requires genuinely holding fewer bytes live.

Modes:
  correctness -> CSPRNG anti-cache probe (two diff inputs same shape; bit-identical outputs =>
                 reject) + all hidden shapes vs the fp32 reference on (y, dx). Emits
                 WRE_RESULT {"mode":"correctness","correctness_ok":bool,...}
  timing      -> median peak_bytes on the primary (large) shape + a per-shape-spread diagnostic
                 across a shape suite. Emits WRE_RESULT {"mode":"timing","timing_ms":<peak_bytes>,...}

Anti-cheat (verifier-patterns.md §B3): every measured iteration regenerates the input `x` with a
fresh seed; the CSPRNG correctness probe runs the op twice on one shape with DIFFERENT inputs
(bit-identical outputs => output-caching cheat => reject) — the hard anti-cache gate. The
per-shape-spread check (memory grows with shape) is a diagnostic only.
"""
import json
import math
import secrets
import sys

import torch

KERNEL_PATH = "/app/repo/submission/kernel.py"
RTOL = 2e-2
ATOL = 2e-2

# Primary shape for the metric (Llama-MLP-ish): large enough that the full [T, I] activations
# dominate the footprint, so materialize-all vs low-resident peak separates sharply.
PRIMARY = {"T": 8192, "H": 4096, "I": 14336, "chunk_size": 512}

# Hidden correctness suite — spans size-1, non-power-of-2, chunk_size dividing / not dividing T,
# single-chunk (chunk_size >= T), and large.
CORRECTNESS_SHAPES = [
    {"T": 1, "H": 128, "I": 256, "chunk_size": 64, "seed": 101},        # size-1 rows
    {"T": 17, "H": 512, "I": 1408, "chunk_size": 8, "seed": 102},        # non-multiple T, chunk remainder
    {"T": 512, "H": 2048, "I": 5632, "chunk_size": 512, "seed": 103},    # chunk_size == T (single chunk)
    {"T": 333, "H": 1600, "I": 4320, "chunk_size": 128, "seed": 104},    # odd/non-pow2 dims
    {"T": 2048, "H": 4096, "I": 14336, "chunk_size": 500, "seed": 105},  # chunk_size does not divide T
    {"T": 8192, "H": 4096, "I": 14336, "chunk_size": 512, "seed": 106},  # large
]

# Shape suite for the anti-flat per-shape-spread diagnostic (peak grows with shape).
SPREAD_SHAPES = [
    {"T": 1024, "H": 2048, "I": 5632, "chunk_size": 256, "seed": 201},
    {"T": 4096, "H": 4096, "I": 14336, "chunk_size": 512, "seed": 202},
    {"T": 8192, "H": 4096, "I": 14336, "chunk_size": 512, "seed": 203},
]

_PER_SHAPE_SPREAD_MIN = 1.5     # peak_bytes must scale with shape (diagnostic only)
_PER_ITER_STABILITY_MAX = 1.5   # peak is deterministic across iters of a shape -> ~1.0 (diagnostic)


def _gen(shape, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    T, H, I = shape["T"], shape["H"], shape["I"]
    cs = int(shape.get("chunk_size", 512))
    x = torch.randn(T, H, device=device, dtype=torch.bfloat16, generator=g) * 0.5
    w_gate = torch.randn(H, I, device=device, dtype=torch.bfloat16, generator=g) * (H ** -0.5)
    w_up = torch.randn(H, I, device=device, dtype=torch.bfloat16, generator=g) * (H ** -0.5)
    w_down = torch.randn(I, H, device=device, dtype=torch.bfloat16, generator=g) * (I ** -0.5)
    grad_out = torch.randn(T, H, device=device, dtype=torch.bfloat16, generator=g) * 0.5
    config = {"T": T, "H": H, "I": I, "chunk_size": cs}
    return (x, w_gate, w_up, w_down, grad_out, config)


def _reference(data):
    """fp32 ground truth from the disclosed contract math (NOT the low-memory oracle).

    Forward + manual input-gradient backward of a SwiGLU gated MLP block. Returns (y, dx)."""
    x, w_gate, w_up, w_down, grad_out, config = data
    xf = x.float(); wg = w_gate.float(); wu = w_up.float(); wd = w_down.float(); go = grad_out.float()
    g = xf @ wg                              # [T, I]
    u = xf @ wu                              # [T, I]
    sig = torch.sigmoid(g)
    silu = g * sig
    a = silu * u                             # [T, I]
    y = a @ wd                               # [T, H]
    da = go @ wd.t()                         # [T, I]
    du = da * silu
    dsilu = da * u
    silup = sig * (1.0 + g * (1.0 - sig))    # silu'(g)
    dg = dsilu * silup
    dx = dg @ wg.t() + du @ wu.t()           # [T, H]
    return y, dx


def _clone(data):
    x, w_gate, w_up, w_down, grad_out, config = data
    return (x.clone(), w_gate.clone(), w_up.clone(), w_down.clone(), grad_out.clone(), dict(config))


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _check(out, ref):
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        return False, f"expected a (y, dx) 2-tuple, got {type(out).__name__}"
    r_y, r_dx = ref
    for name, o, r in (("y", out[0], r_y), ("dx", out[1], r_dx)):
        o = o.float(); r = r.float()
        if o.shape != r.shape:
            return False, f"{name} shape {tuple(o.shape)} != {tuple(r.shape)}"
        if not torch.isfinite(o).all():
            return False, f"{name} has non-finite values"
        diff = (o - r).abs()
        tol = ATOL + RTOL * r.abs()
        nbad = int((diff > tol).sum())
        if nbad > 0:
            return False, f"{name}: {nbad} elems exceed rtol=atol={RTOL} (max {float(diff.max()):.4g})"
    return True, "ok"


# total visible correctness cases = the CSPRNG anti-cache probe
# (one case) + every hidden shape. Reported as tests{passed,total} in the verdict so
# the reward.md implementation-class contract ("EVERY case must pass") is auditable.
N_CORRECTNESS_CASES = 1 + len(CORRECTNESS_SHAPES)


def run_correctness(custom_kernel):
    # `passed` counts cases cleared so far; the suite is fail-fast, so on the first
    # failure `passed` is exactly the number of cases that did pass.
    passed = 0
    # Stage 0 — CSPRNG anti-cache probe: two DIFFERENT inputs, same shape.
    probe_shape = {"T": 256, "H": 2048, "I": 5632, "chunk_size": 128}
    sa, sb = secrets.randbits(62), secrets.randbits(62)
    da_, db_ = _gen(probe_shape, sa), _gen(probe_shape, sb)
    oa = custom_kernel(_clone(da_)); torch.cuda.synchronize()
    oa = tuple(t.detach().clone() for t in oa)
    ob = custom_kernel(_clone(db_)); torch.cuda.synchronize()
    ob = tuple(t.detach().clone() for t in ob)
    ok_a, msg_a = _check(oa, _reference(da_))
    if not ok_a:
        return False, f"cache-probe: first call wrong: {msg_a}", passed
    ok_b, msg_b = _check(ob, _reference(db_))
    if not ok_b:
        return False, f"cache-probe: second call wrong (output caching?): {msg_b}", passed
    if all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(oa, ob)):
        return False, "cache-probe: bit-identical outputs for different inputs (caching cheat)", passed
    passed += 1                                  # anti-cache probe cleared
    del oa, ob, da_, db_
    torch.cuda.empty_cache()
    # Stage 1 — all hidden shapes (y AND dx).
    for spec in CORRECTNESS_SHAPES:
        shape = {k: spec[k] for k in ("T", "H", "I", "chunk_size")}
        data = _gen(shape, spec["seed"])
        out = custom_kernel(_clone(data)); torch.cuda.synchronize()
        ok, msg = _check(out, _reference(data))
        if not ok:
            return False, f"shape {shape} seed {spec['seed']}: {msg}", passed
        passed += 1
        del data, out
        torch.cuda.empty_cache()
    return True, "all correctness passed", passed


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _peak_shape(custom_kernel, shape, seed, iters=3):
    """Median peak GPU bytes of a custom_kernel call. Weights/grad_out are the fixed persistent
    inputs (allocated before the peak counter is reset, so the peak reflects the intermediates the
    call holds); x is regenerated fresh each measured iteration."""
    base = _gen(shape, seed)
    _, wg, wu, wd, go, config = base
    # warmup: CUDA context + cuBLAS workspace, distinct input content from the timed iters
    warm = _gen(shape, seed - 1)[0]
    for _ in range(2):
        out = custom_kernel((warm, wg, wu, wd, go, config))
    torch.cuda.synchronize()
    del out, warm
    peaks = []
    for i in range(iters):
        xi = torch.randn(shape["T"], shape["H"], device="cuda", dtype=torch.bfloat16,
                         generator=torch.Generator(device="cuda").manual_seed(seed + 1000 + i)) * 0.5
        data = (xi, wg, wu, wd, go, config)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        out = custom_kernel(data)
        torch.cuda.synchronize()
        peaks.append(int(torch.cuda.max_memory_allocated()))
        del out, data, xi
    med = _median([float(p) for p in peaks])
    hi = max(peaks); lo = min(peaks)
    per_iter = (hi / lo) if lo > 0 else float("inf")   # deterministic memory -> ~1.0
    return med, per_iter


def run_timing(custom_kernel):
    med, stab = _peak_shape(custom_kernel, PRIMARY, seed=203)
    spread = []
    for spec in SPREAD_SHAPES:
        shape = {k: spec[k] for k in ("T", "H", "I", "chunk_size")}
        m, _ = _peak_shape(custom_kernel, shape, spec["seed"])
        spread.append(m)
    spread_ratio = max(spread) / min(spread) if min(spread) > 0 else float("inf")
    return med, stab, spread_ratio


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRE_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    torch.cuda.synchronize()
    try:
        custom_kernel = _load_kernel()
    except Exception as exc:
        print("WRE_RESULT " + json.dumps({"mode": mode, "correctness_ok": False,
              "error": f"load_failed: {type(exc).__name__}: {exc}"})); sys.exit(3)

    if mode == "correctness":
        try:
            ok, msg, passed = run_correctness(custom_kernel)
        except NotImplementedError as exc:
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "cases_passed": 0, "cases_total": N_CORRECTNESS_CASES,
                  "error": f"not_implemented: {exc}"})); sys.exit(3)
        except Exception as exc:
            import traceback
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "cases_passed": 0, "cases_total": N_CORRECTNESS_CASES,
                  "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]})); sys.exit(3)
        print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": bool(ok),
              "cases_passed": int(passed), "cases_total": N_CORRECTNESS_CASES, "detail": msg}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        try:
            med, stab, spread = run_timing(custom_kernel)
        except Exception as exc:
            import traceback
            print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]})); sys.exit(3)
        flat_ok = spread >= _PER_SHAPE_SPREAD_MIN
        stable_ok = stab <= _PER_ITER_STABILITY_MAX
        print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": med,
              "per_iter_max_min": stab, "per_shape_spread": spread,
              "flat_ok": bool(flat_ok), "stable_ok": bool(stable_ok),
              "primary": PRIMARY}))
        # peak_bytes metric: flat_ok/stable_ok are diagnostics; the CSPRNG cache-probe is the hard
        # anti-cache gate. Only an invalid measurement (peak<=0 / non-finite) hard-fails timing.
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
