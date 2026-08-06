#!/usr/bin/env python3
"""Verifier workload for wre-tp-allreduce-bias-fuse.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden tensor-parallel
row-parallel "all-reduce + bias" combine workloads: reduce R rank-partial [T, D] outputs over the
rank axis and add the bias once. Correctness = match a SEEDED fp32 reference within rtol=atol=2e-2,
computed independently here (NOT the oracle, which is never baked into the image).

Modes:
  correctness -> CSPRNG anti-cache probe + all hidden shapes vs the fp32 reference.
  timing      -> block-of-medians CUDA-event latency on the primary shape + a per-shape-spread
                 diagnostic. Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}
"""
import json
import math
import secrets
import sys

import torch

KERNEL_PATH = "/app/repo/submission/kernel.py"
RTOL = 2e-2
ATOL = 2e-2

PRIMARY = {"R": 16, "T": 4096, "D": 4096}

CORRECTNESS_SHAPES = [
    {"R": 2, "T": 1, "D": 128, "seed": 101},      # single row
    {"R": 3, "T": 17, "D": 8191, "seed": 102},    # odd T, non-pow2 D
    {"R": 8, "T": 512, "D": 2048, "seed": 103},
    {"R": 5, "T": 333, "D": 1, "seed": 104},      # D=1 edge
    {"R": 16, "T": 1024, "D": 4096, "seed": 105},
    {"R": 1, "T": 256, "D": 512, "seed": 106},    # R=1 (reduce of a single rank -> just +bias)
]

SPREAD_SHAPES = [
    {"R": 4, "T": 512, "D": 1024, "seed": 201},
    {"R": 8, "T": 2048, "D": 2048, "seed": 202},
    {"R": 16, "T": 4096, "D": 4096, "seed": 203},
]

_PER_SHAPE_SPREAD_MIN = 3.0
_LATENCY_STABILITY_MAX = 8.0
_STABILITY_NOISE_FLOOR_MS = 0.12


def _gen(shape, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    R, T, D = shape["R"], shape["T"], shape["D"]
    partials = torch.randn(R, T, D, device=device, dtype=torch.bfloat16, generator=g) * 0.5
    bias = torch.randn(D, device=device, dtype=torch.bfloat16, generator=g) * 0.5
    return (partials, bias, {"R": R, "T": T, "D": D})


def _reference(data):
    """fp32 ground truth: sum the partials over the rank axis, add the bias once (the contract)."""
    partials, bias, config = data
    acc = partials.float().sum(dim=0)
    acc = acc + bias.float()
    return acc.to(torch.bfloat16)


def _clone(data):
    partials, bias, config = data
    return (partials.clone(), bias.clone(), dict(config))


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
    probe = {"R": 8, "T": 256, "D": 4096}
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
        shape = {k: spec[k] for k in ("R", "T", "D")}
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
    _, _, config = base
    warm = _gen(shape, seed - 1)
    for _ in range(10):
        custom_kernel(_clone(warm))
    torch.cuda.synchronize()
    durs = []
    for i in range(60):
        data = _gen(shape, seed + 1000 + i)
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
    med, stab = _time_shape(custom_kernel, PRIMARY, seed=203)
    spread = []
    for spec in SPREAD_SHAPES:
        shape = {k: spec[k] for k in ("R", "T", "D")}
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
