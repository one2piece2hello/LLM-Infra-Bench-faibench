#!/usr/bin/env python3
"""Verifier workload for mamba-zoh-discretize. GPU task (anchor calibrated on H20).

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden Mamba selective-scan
ZOH-discretization workloads. Correctness = match a SEEDED fp32 reference (the disclosed contract
math, re-derived here vectorized; NOT the oracle) within rtol=atol=1e-3.

Modes:
  correctness -> CSPRNG anti-cache probe (two distinct input draws -> distinct outputs) +
                 hidden shapes (L=1, odd/non-pow2 L/D/N, larger) vs the reference.
  timing      -> block-of-medians CUDA-event latency on the primary shape (a sequence, where a
                 per-timestep Python loop launches many tiny kernels vs one fused vectorized pass)
                 + a per-shape-spread diagnostic. Emits WRE_RESULT {"mode":"timing","timing_ms":..}.
"""
import json
import math
import secrets
import sys

import torch

KERNEL_PATH = "/app/repo/submission/kernel.py"
RTOL = 1e-3
ATOL = 1e-3

PRIMARY = {"Bt": 16, "L": 64, "D": 512, "N": 16}

CORRECTNESS_SHAPES = [
    {"Bt": 1, "L": 1, "D": 4, "N": 2, "seed": 101},        # L=1 edge
    {"Bt": 2, "L": 17, "D": 5, "N": 3, "seed": 102},       # odd L, small
    {"Bt": 3, "L": 64, "D": 32, "N": 16, "seed": 103},
    {"Bt": 2, "L": 100, "D": 24, "N": 7, "seed": 104},     # odd/non-pow2
    {"Bt": 4, "L": 128, "D": 64, "N": 16, "seed": 105},
    {"Bt": 1, "L": 33, "D": 17, "N": 4, "seed": 106},      # odd everything
]

SPREAD_SHAPES = [
    {"Bt": 8, "L": 32, "D": 512, "N": 16, "seed": 201},
    {"Bt": 8, "L": 64, "D": 512, "N": 16, "seed": 202},
    {"Bt": 8, "L": 96, "D": 512, "N": 16, "seed": 203},
]

_PER_SHAPE_SPREAD_MIN = 1.5
_LATENCY_STABILITY_MAX = 8.0
_STABILITY_NOISE_FLOOR_MS = 0.12


def _gen(shape, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    Bt, L, D, N = shape["Bt"], shape["L"], shape["D"], shape["N"]
    u = torch.randn(Bt, L, D, device=device, dtype=torch.float32, generator=g)
    delta = torch.rand(Bt, L, D, device=device, dtype=torch.float32, generator=g) * 0.09 + 0.01  # positive [0.01,0.1]
    A = -torch.exp(torch.randn(D, N, device=device, dtype=torch.float32, generator=g))            # negative state matrix
    B = torch.randn(Bt, L, N, device=device, dtype=torch.float32, generator=g)
    return (u, delta, A, B)


def _reference(data):
    u, delta, A, B = data
    deltaA = torch.exp(delta.unsqueeze(-1) * A[None, None, :, :])
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    return (deltaA, deltaB_u)


def _clone(data):
    u, delta, A, B = data
    return (u.clone(), delta.clone(), A.clone(), B.clone())


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _check(out, ref):
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        return False, f"output must be a (deltaA, deltaB_u) tuple, got {type(out)}"
    names = ("deltaA", "deltaB_u")
    for i in range(2):
        o = out[i]
        if not torch.is_tensor(o):
            return False, f"{names[i]} is not a tensor: {type(o)}"
        o = o.float(); r = ref[i].float()
        if o.shape != r.shape:
            return False, f"{names[i]} shape {tuple(o.shape)} != {tuple(r.shape)}"
        if not torch.isfinite(o).all():
            return False, f"{names[i]} has non-finite values"
        diff = (o - r).abs(); tol = ATOL + RTOL * r.abs()
        nbad = int((diff > tol).sum())
        if nbad > 0:
            return False, f"{names[i]} {nbad} elems exceed rtol=atol={RTOL} (max {float(diff.max()):.4g})"
    return True, "ok"


def run_correctness(custom_kernel):
    probe = {"Bt": 4, "L": 32, "D": 48, "N": 8}
    da, db = _gen(probe, secrets.randbits(62)), _gen(probe, secrets.randbits(62))
    oa = custom_kernel(_clone(da)); torch.cuda.synchronize()
    oa = tuple(t.detach().clone() for t in oa)
    ob = custom_kernel(_clone(db)); torch.cuda.synchronize()
    ob = tuple(t.detach().clone() for t in ob)
    ok_a, msg_a = _check(oa, _reference(da))
    if not ok_a:
        return False, f"cache-probe: first call wrong: {msg_a}"
    ok_b, msg_b = _check(ob, _reference(db))
    if not ok_b:
        return False, f"cache-probe: second call wrong (output caching?): {msg_b}"
    if oa[0].shape == ob[0].shape and torch.equal(oa[0], ob[0]):
        return False, "cache-probe: bit-identical deltaA for different inputs (caching cheat)"
    for spec in CORRECTNESS_SHAPES:
        shape = {k: spec[k] for k in ("Bt", "L", "D", "N")}
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
    _, delta, A, B = base
    Bt, L, D, N = shape["Bt"], shape["L"], shape["D"], shape["N"]
    warm = _gen(shape, seed - 1)[0]
    for _ in range(10):
        custom_kernel((warm.clone(), delta.clone(), A.clone(), B.clone()))
    torch.cuda.synchronize()
    durs = []
    for i in range(50):
        g = torch.Generator(device="cuda").manual_seed(seed + 1000 + i)
        ui = torch.randn(Bt, L, D, device="cuda", dtype=torch.float32, generator=g)
        data = (ui, delta, A, B)
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
        shape = {k: spec[k] for k in ("Bt", "L", "D", "N")}
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
