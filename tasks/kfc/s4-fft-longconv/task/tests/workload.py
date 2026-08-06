#!/usr/bin/env python3
"""Verifier workload for s4-fft-longconv. GPU task (anchor calibrated on H20).

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden causal long-conv
workloads. Correctness = match a SEEDED fp32 reference (the direct causal-convolution definition,
re-derived here as a lower-triangular Toeplitz apply, NOT the FFT oracle) within rtol=atol=2e-2.

Modes:
  correctness -> CSPRNG anti-cache probe + hidden shapes (L=1 edge, odd/non-pow2 L, large L) vs ref.
  timing      -> block-of-medians CUDA-event latency on the primary shape (long L, where an O(L^2)
                 direct/Toeplitz apply is far slower than an O(L log L) FFT) + a per-shape-spread
                 diagnostic. Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys

import torch

KERNEL_PATH = "/app/repo/submission/kernel.py"
RTOL = 2e-2
ATOL = 2e-2

PRIMARY = {"B": 16, "H": 16, "L": 1024}

CORRECTNESS_SHAPES = [
    {"B": 1, "H": 1, "L": 1, "seed": 101},        # L=1 edge (y = k[:,0]*u[:,:,0])
    {"B": 1, "H": 1, "L": 2, "seed": 102},        # tiny
    {"B": 2, "H": 3, "L": 17, "seed": 103},       # odd L
    {"B": 2, "H": 2, "L": 64, "seed": 104},
    {"B": 4, "H": 4, "L": 255, "seed": 105},      # non-pow2 L
    {"B": 3, "H": 5, "L": 100, "seed": 106},      # odd L, odd H
    {"B": 2, "H": 2, "L": 1024, "seed": 107},     # large L
]

SPREAD_SHAPES = [
    {"B": 8, "H": 8, "L": 512, "seed": 201},
    {"B": 8, "H": 8, "L": 1024, "seed": 202},
    {"B": 8, "H": 8, "L": 2048, "seed": 203},
]

_PER_SHAPE_SPREAD_MIN = 1.5
_LATENCY_STABILITY_MAX = 8.0
_STABILITY_NOISE_FLOOR_MS = 0.12


def _gen(shape, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, L = shape["B"], shape["H"], shape["L"]
    u = torch.randn(B, H, L, device=device, dtype=torch.bfloat16, generator=g)
    # a decaying (stable) SSM-style kernel: randn * exp(-l/tau), tau ~ L/8, so outputs stay O(1).
    l = torch.arange(L, device=device, dtype=torch.float32)
    decay = torch.exp(-l / max(1.0, L / 8.0))
    k = (torch.randn(H, L, device=device, dtype=torch.float32, generator=g) * decay).to(torch.bfloat16)
    return (u, k, {"L": L})


def _reference(data):
    """fp32 ground truth: direct causal convolution (lower-triangular Toeplitz apply)."""
    u, k, config = data
    B, H, L = u.shape
    uf = u.float()
    kf = k.float()
    idx = torch.arange(L, device=u.device)
    d = idx[:, None] - idx[None, :]
    mask = (d >= 0)
    dc = d.clamp(min=0)
    T = kf[:, dc] * mask                      # [H, L, L]
    return torch.einsum("h t s, b h s -> b h t", T, uf)


def _clone(data):
    u, k, config = data
    return (u.clone(), k.clone(), dict(config))


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _check(out, ref):
    if not torch.is_tensor(out):
        return False, f"output is not a tensor: {type(out)}"
    o = out.float()
    if o.shape != ref.shape:
        return False, f"shape {tuple(o.shape)} != {tuple(ref.shape)}"
    if not torch.isfinite(o).all():
        return False, "output has non-finite values"
    diff = (o - ref).abs()
    tol = ATOL + RTOL * ref.abs()
    nbad = int((diff > tol).sum())
    if nbad > 0:
        return False, f"{nbad} elems exceed rtol=atol={RTOL} (max {float(diff.max()):.4g})"
    return True, "ok"


def run_correctness(custom_kernel):
    probe = {"B": 4, "H": 4, "L": 128}
    da, db = _gen(probe, secrets.randbits(62)), _gen(probe, secrets.randbits(62))
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
    for spec in CORRECTNESS_SHAPES:
        shape = {k: spec[k] for k in ("B", "H", "L")}
        data = _gen(shape, spec["seed"])
        out = custom_kernel(_clone(data)); torch.cuda.synchronize()
        ok, msg = _check(out, _reference(data))
        if not ok:
            return False, f"shape {shape} seed {spec['seed']}: {msg}"
    return True, "all correctness passed"


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _time_shape(custom_kernel, shape, seed):
    base = _gen(shape, seed)
    _, k, config = base
    warm = _gen(shape, seed - 1)[0]
    for _ in range(10):
        custom_kernel((warm.clone(), k.clone(), dict(config)))
    torch.cuda.synchronize()
    B, H, L = shape["B"], shape["H"], shape["L"]
    durs = []
    for i in range(60):
        g = torch.Generator(device="cuda").manual_seed(seed + 1000 + i)
        ui = torch.randn(B, H, L, device="cuda", dtype=torch.bfloat16, generator=g)
        data = (ui, k, dict(config))
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); custom_kernel(data); e.record(); torch.cuda.synchronize()
        durs.append(s.elapsed_time(e))
    med = _median(durs)
    ss = sorted(durs); n = len(ss)
    p5 = ss[max(0, n // 20)]; p95 = ss[min(n - 1, n - 1 - n // 20)]
    ratio = (p95 / p5) if p5 > 0 else float("inf")
    return med, ratio


def run_timing(custom_kernel):
    med, stab = _time_shape(custom_kernel, PRIMARY, seed=202)
    spread = []
    for spec in SPREAD_SHAPES:
        shape = {k: spec[k] for k in ("B", "H", "L")}
        m, _ = _time_shape(custom_kernel, shape, spec["seed"])
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
              "flat_ok": bool(flat_ok), "stable_ok": bool(stable_ok), "primary": PRIMARY}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
