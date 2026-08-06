#!/usr/bin/env python3
"""Verifier workload for wre-stability-fused-gradclip.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` clipping a list of gradient
tensors by their GLOBAL L2 norm. Correctness = match a SEEDED reference (original global norm +
globally-scaled gradients), computed INDEPENDENTLY here (never the oracle), within rtol=atol=2e-3.

Modes:
  correctness -> CSPRNG anti-cache probe + hidden shapes (clip and no-clip cases) vs the reference.
  timing      -> CUDA-event block-of-medians over a large param list. Emits
                 WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys

import torch

KERNEL_PATH = "/app/repo/submission/kernel.py"
RTOL = 2e-3
ATOL = 2e-3

torch.backends.cuda.matmul.allow_tf32 = False


def _make_grads(shapes, seed, scale, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(*s, device=device, dtype=torch.float32, generator=g) * scale for s in shapes]


def _reference(grads, max_norm):
    sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for t in grads:
        sq = sq + (t.float() * t.float()).sum()
    total_norm = torch.sqrt(sq)
    coef = max_norm / (total_norm + 1e-6)
    if float(coef) < 1.0:
        out = [t.float() * coef for t in grads]
    else:
        out = [t.float().clone() for t in grads]
    return out, total_norm


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _check(ret, ref):
    if not (isinstance(ret, (tuple, list)) and len(ret) == 2):
        return False, f"output must be (grads, total_norm), got {type(ret)}"
    gout, nout = ret
    gref, nref = ref
    if not torch.is_tensor(nout):
        return False, "total_norm must be a tensor"
    if abs(float(nout) - float(nref)) > (ATOL + RTOL * abs(float(nref))):
        return False, f"total_norm {float(nout):.5g} != {float(nref):.5g}"
    if len(gout) != len(gref):
        return False, f"grad list length {len(gout)} != {len(gref)}"
    for i, (a, b) in enumerate(zip(gout, gref)):
        a = a.float()
        if a.shape != b.shape:
            return False, f"grad {i} shape {tuple(a.shape)} != {tuple(b.shape)}"
        if not torch.isfinite(a).all():
            return False, f"grad {i} non-finite"
        diff = (a - b).abs(); tol = ATOL + RTOL * b.abs()
        if int((diff > tol).sum()) > 0:
            return False, f"grad {i} mismatch (max {float(diff.max()):.4g})"
    return True, "ok"


CORRECTNESS_CASES = [
    # (shapes, scale, max_norm, seed) ; large scale -> clip triggers, small scale -> no clip
    ([(64,), (128, 32), (16, 16)], 1.0, 1.0, 101),        # clip
    ([(256, 128), (512,), (32, 8)], 2.0, 0.5, 102),        # clip hard
    ([(10,), (20,)], 0.001, 100.0, 103),                    # no clip (tiny grads)
    ([(333,), (129, 7)], 1.5, 2.0, 104),                    # odd dims, clip
    ([(1024,)], 0.5, 3.0, 105),                             # single tensor, borderline
]


def run_correctness(custom_kernel):
    shp = [(128, 64), (256,), (32, 32)]
    sa = _make_grads(shp, secrets.randbits(62), 1.5)
    sb = _make_grads(shp, secrets.randbits(62), 1.5)
    ra = _reference([t.clone() for t in sa], 1.0)
    rb = _reference([t.clone() for t in sb], 1.0)
    oa = custom_kernel(([t.clone() for t in sa], {"max_norm": 1.0})); torch.cuda.synchronize()
    ob = custom_kernel(([t.clone() for t in sb], {"max_norm": 1.0})); torch.cuda.synchronize()
    ok, msg = _check(oa, ra)
    if not ok:
        return False, f"cache-probe A wrong: {msg}"
    ok, msg = _check(ob, rb)
    if not ok:
        return False, f"cache-probe B wrong (caching?): {msg}"
    if abs(float(oa[1]) - float(ob[1])) < 1e-6:
        return False, "cache-probe: identical total_norm for different inputs (caching cheat)"
    for (shapes, scale, max_norm, seed) in CORRECTNESS_CASES:
        base = _make_grads(shapes, seed, scale)
        ref = _reference([t.clone() for t in base], max_norm)
        out = custom_kernel(([t.clone() for t in base], {"max_norm": max_norm})); torch.cuda.synchronize()
        ok, msg = _check(out, ref)
        if not ok:
            return False, f"case seed {seed} scale {scale} max_norm {max_norm}: {msg}"
    return True, "all correctness passed"


TIMING_N = 256
TIMING_SHAPE = (256, 256)
TIMING_MAXNORM = 1.0     # small -> clip always triggers (both variants do the scale pass)
TIMING_SCALE = 1.0


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def run_timing(custom_kernel):
    shapes = [TIMING_SHAPE] * TIMING_N
    for w in range(4):
        custom_kernel((_make_grads(shapes, 900 + w, TIMING_SCALE), {"max_norm": TIMING_MAXNORM}))
    torch.cuda.synchronize()
    durs = []
    for i in range(30):
        grads = _make_grads(shapes, 1000 + i, TIMING_SCALE)   # fresh (mutated in place) — outside timing
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); custom_kernel((grads, {"max_norm": TIMING_MAXNORM})); e.record(); torch.cuda.synchronize()
        durs.append(s.elapsed_time(e))
    med = _median(durs)
    ss = sorted(durs); n = len(ss)
    p5 = ss[max(0, n // 20)]; p95 = ss[min(n - 1, n - 1 - n // 20)]
    stab = (p95 / p5) if p5 > 0 else float("inf")
    return med, stab, stab


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRE_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
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
        print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": med,
              "per_iter_max_min": stab, "per_shape_spread": spread,
              "flat_ok": True, "stable_ok": True, "primary": {"N": TIMING_N, "shape": list(TIMING_SHAPE)}}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
