#!/usr/bin/env python3
"""Verifier workload for wre-parallel-all2all-redistribute.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden block-partitioned
bf16 CUDA tensors x [world_size, world_size, chunk, D]. Correctness = match a SEEDED fp32 reference
within rtol=atol=2e-2 (the reference IS the disclosed contract: y[d,s] = x[s,d], a swap of the two
leading rank axes, materialized contiguous — NOT the fast oracle, which is never baked into the image). The
verifier observes only the redistributed tensor, so a naive per-(src,dst)-block copy loop is CORRECT
but slow, and only the single-transpose coalescing earns the speedup.

Modes:
  correctness -> CSPRNG anti-cache probe (two DIFFERENT inputs, same shape) + all hidden shapes vs
                 the fp32 reference (incl. shapes with chunk != world_size, which catch a wrong-axis
                 transpose by shape). Emits WRE_RESULT {"mode":"correctness","correctness_ok":bool,...}
  timing      -> block-of-medians CUDA-event latency on the primary shape (x regenerated with a fresh
                 seed every timed iteration) + a per-shape-spread anti-flat diagnostic across world_size.
                 Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}

Anti-cheat (verifier-patterns.md §B3): every timed iteration regenerates x with a fresh seed; the
CSPRNG probe runs the op twice on one shape with DIFFERENT inputs (bit-identical outputs =>
output-caching cheat => reject). flat_ok/stable_ok are RECORDED diagnostics only.
"""
import json
import math
import secrets
import sys

import torch

KERNEL_PATH = "/app/repo/submission/kernel.py"
RTOL = 2e-2
ATOL = 2e-2

# Primary timing shape: many ranks so the O(W^2) per-block copy loop (baseline2) dwarfs the single
# transpose+contiguous (oracle). W=64 -> 4096 block copies vs 1 coalesced kernel. chunk*D modest so
# the score is launch-overhead-bound, and the coalescing win grows with W.
PRIMARY = {"world_size": 64, "chunk": 8, "D": 128}

CORRECTNESS_SHAPES = [
    {"world_size": 1, "chunk": 1, "D": 4, "seed": 101},        # single rank, size-1
    {"world_size": 2, "chunk": 3, "D": 8, "seed": 102},        # chunk != W (wrong-axis => shape fail)
    {"world_size": 5, "chunk": 5, "D": 16, "seed": 103},       # chunk == W (wrong-axis => value fail)
    {"world_size": 8, "chunk": 7, "D": 40, "seed": 104},       # non-pow2 chunk/D
    {"world_size": 16, "chunk": 4, "D": 64, "seed": 105},
    {"world_size": 64, "chunk": 8, "D": 128, "seed": 106},     # large (== primary)
]

SPREAD_SHAPES = [
    {"world_size": 8, "chunk": 8, "D": 128, "seed": 201},
    {"world_size": 32, "chunk": 8, "D": 128, "seed": 202},
    {"world_size": 64, "chunk": 8, "D": 128, "seed": 203},
]
_PER_SHAPE_SPREAD_MIN = 2.0
_LATENCY_STABILITY_MAX = 8.0
_STABILITY_NOISE_FLOOR_MS = 0.12


def _gen(spec, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    W, chunk, D = spec["world_size"], spec["chunk"], spec["D"]
    x = (torch.randn(W, W, chunk, D, device=device, dtype=torch.bfloat16, generator=g) * 0.5)
    config = {"world_size": W, "chunk": chunk, "D": D}
    return (x, config)


def _reference(data):
    """fp32 ground truth: y[d,s] = x[s,d] (swap the two leading rank axes), contiguous."""
    x, config = data
    return x.float().transpose(0, 1).contiguous()


def _clone(data):
    x, config = data
    return (x.clone(), dict(config))


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _check(out, ref):
    if not torch.is_tensor(out):
        return False, f"output is {type(out).__name__}, expected a tensor"
    o = out.float()
    r = ref.float()
    if tuple(o.shape) != tuple(r.shape):
        return False, f"shape {tuple(o.shape)} != {tuple(r.shape)} (wrong transpose axes?)"
    if not torch.isfinite(o).all():
        return False, "non-finite values"
    diff = (o - r).abs()
    tol = ATOL + RTOL * r.abs()
    nbad = int((diff > tol).sum())
    if nbad > 0:
        return False, f"{nbad} elems exceed rtol=atol={RTOL} (max {float(diff.max()):.4g})"
    return True, "ok"


def run_correctness(custom_kernel):
    # Stage 0 — CSPRNG anti-cache probe: two DIFFERENT inputs, same shape.
    probe = {"world_size": 8, "chunk": 8, "D": 64}
    sa, sb = secrets.randbits(62), secrets.randbits(62)
    da, db = _gen(probe, sa), _gen(probe, sb)
    oa = custom_kernel(_clone(da)); torch.cuda.synchronize(); oa = oa.detach().clone()
    ob = custom_kernel(_clone(db)); torch.cuda.synchronize(); ob = ob.detach().clone()
    ok_a, msg_a = _check(oa, _reference(da))
    if not ok_a:
        return False, f"cache-probe: first call wrong: {msg_a}"
    ok_b, msg_b = _check(ob, _reference(db))
    if not ok_b:
        return False, f"cache-probe: second call wrong (output caching?): {msg_b}"
    if oa.shape == ob.shape and torch.equal(oa, ob):
        return False, "cache-probe: bit-identical outputs for different inputs (caching cheat)"
    # Stage 1 — all hidden shapes.
    for spec in CORRECTNESS_SHAPES:
        data = _gen(spec, spec["seed"])
        out = custom_kernel(_clone(data)); torch.cuda.synchronize()
        ok, msg = _check(out, _reference(data))
        if not ok:
            return False, f"W={spec['world_size']} chunk={spec['chunk']} D={spec['D']} seed={spec['seed']}: {msg}"
    return True, "all correctness passed"


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _time_shape(custom_kernel, spec, seed):
    warm = _gen(spec, seed - 1)
    for _ in range(8):
        custom_kernel(_clone(warm))
    torch.cuda.synchronize()
    durs = []
    for i in range(50):
        data = _gen(spec, seed + 1000 + i)      # regenerate x every iter (OUTSIDE timing)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); custom_kernel(data); e.record(); torch.cuda.synchronize()
        durs.append(s.elapsed_time(e))  # ms
    med = _median(durs)
    ss = sorted(durs); n = len(ss)
    p5 = ss[max(0, n // 20)]; p95 = ss[min(n - 1, n - 1 - n // 20)]
    ratio = (p95 / p5) if p5 > 0 else float("inf")
    return med, ratio


def run_timing(custom_kernel):
    med, stab = _time_shape(custom_kernel, PRIMARY, seed=203)
    spread = []
    for spec in SPREAD_SHAPES:
        m, _ = _time_shape(custom_kernel, spec, spec["seed"])
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
            ok, msg = run_correctness(custom_kernel)
        except NotImplementedError as exc:
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"not_implemented: {exc}"})); sys.exit(3)
        except Exception as exc:
            import traceback
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]})); sys.exit(3)
        print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": bool(ok), "detail": msg}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        try:
            med, stab, spread = run_timing(custom_kernel)
        except Exception as exc:
            print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "error": f"{type(exc).__name__}: {exc}"})); sys.exit(3)
        flat_ok = spread >= _PER_SHAPE_SPREAD_MIN
        stable_ok = (stab <= _LATENCY_STABILITY_MAX) or (med < _STABILITY_NOISE_FLOOR_MS)
        print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": med,
              "per_iter_max_min": stab, "per_shape_spread": spread,
              "flat_ok": bool(flat_ok), "stable_ok": bool(stable_ok),
              "primary": PRIMARY}))
        valid = (med > 0) and math.isfinite(med)  # flat_ok/stable_ok diagnostic; CSPRNG probe is the hard gate
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
