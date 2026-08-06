#!/usr/bin/env python3
"""Standalone workload for the group-quantised int8-weight / fp16-activation matmul
subsystem (``w8a16.w8a16_matmul``).

Drives the PUBLIC entry ``w8a16_matmul(a, qweight, scales, zeros, group_size)``
(imported from the baked /app/repo tree) with synthetic fixed-seed tensors on the GPU.
The subsystem returns ``a @ dequant(qweight, scales, zeros, group_size)`` where
``a`` is ``[M, K]`` fp16, ``qweight`` is ``[K, N]`` int8 (ASYMMETRIC group-quantised
weight), ``scales`` / ``zeros`` are ``[K // group_size, N]`` (fp16 / int8),
``W[k, n] = (qweight[k, n] - zeros[g, n]) * scales[g, n]`` with ``g = k // group_size``,
and the result is ``a @ W`` reduced with an fp32 accumulator, returned fp16 ``[M, N]``.
The benchmark drives a small-M, large-K/N (weight-heavy) shape: it is memory-bound on
the weight stream.

Two modes:
  correctness : run the subsystem over a DIVERSE hidden suite of (M, K, N, group_size)
                and compare each against an INDEPENDENT fp32 reference computed here
                (NOT part of the editable scope), by relative-norm tolerance. Emits a
                pass-FRACTION over the whole suite (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on the big
                weight-heavy regime; also reports sol_fraction against the H20 roofline.

Emits one line ``WRO_W8A16_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from w8a16 import w8a16_matmul

# SOL helper (pure-math paths are import-safe anywhere; ship alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

# ---- timed regime: small-M, large-K/N (weight-heavy, memory-bound) ----
M = 64
K = 4096
N = 8192
GROUP_SIZE = 128
DTYPE = torch.float16
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
WARMUP = 10
ITERS = 30

# ---- hidden correctness suite: many diverse (M, K, N, group_size) shapes ----
# group_size divides K; the kernel must reload the correct per-group (scale, zero) for
# every K-tile and subtract the integer zero-point on every group. A kernel that only
# handles the timed shape / a single group / forgets the zero-point fails these.
CORR_CASES = [
    (32, 512, 512, 128),      # small
    (48, 768, 1024, 64),      # non-power-of-2 M, small group
    (64, 1024, 1024, 128),    # square-ish
    (128, 2048, 512, 256),    # big group
    (96, 512, 4096, 128),     # wide N
    (64, 2048, 320, 64),      # skewed N
    (33, 1024, 4097, 128),    # ragged M and N (N not multiple of tile)
    (256, 4096, 128, 128),    # skinny N
    (16, 4096, 4096, 256),    # large
    (64, 1536, 1536, 32),     # tiny group (many groups)
    (64, 1024, 768, 512),     # one big group spanning many K-tiles per... (K/group=2)
    (80, 640, 640, 128),      # non-tile-multiple M
    (64, 3072, 1024, 128),    # tall K
    (32, 896, 896, 64),       # prime-ish dims
]


def build_inputs(m, k, n, gs, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, dtype=DTYPE, generator=g) * 0.05
    G = k // gs
    qweight = torch.randint(-128, 128, (k, n), dtype=torch.int8, device=device, generator=g)
    scales = (torch.rand((G, n), device=device, dtype=DTYPE, generator=g) * 0.02 + 1e-3)
    zeros = torch.randint(-8, 8, (G, n), dtype=torch.int8, device=device, generator=g)
    return a, qweight, scales, zeros, gs


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def _ref(a, qweight, scales, zeros, gs):
    G, N = scales.shape
    K = qweight.shape[0]
    s_full = scales.to(torch.float32).repeat_interleave(gs, dim=0)
    z_full = zeros.to(torch.float32).repeat_interleave(gs, dim=0)
    w = (qweight.to(torch.float32) - z_full) * s_full
    return torch.matmul(a.to(torch.float32), w)


def correctness():
    torch.cuda.synchronize()
    n_pass = 0
    detail = {}
    for i, (m, k, n, gs) in enumerate(CORR_CASES):
        a, qw, sc, zr, g = build_inputs(m, k, n, gs, seed=i)
        key = f"{m}x{k}x{n}_g{gs}"
        try:
            o = w8a16_matmul(a, qw, sc, zr, g).float()
        except Exception as e:
            detail[key] = {"error": str(e)[:80], "passed": False}
            continue
        ref = _ref(a, qw, sc, zr, g)
        torch.cuda.synchronize()
        if list(o.shape) != [m, n]:
            detail[key] = {"shape": list(o.shape), "passed": False}
            continue
        rel_max, rel_l2, _ = _relnorm(o, ref)
        passed = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        n_pass += int(passed)
        detail[key] = {"rel_max": round(rel_max, 5), "rel_l2": round(rel_l2, 5), "passed": passed}
    total = len(CORR_CASES)
    frac = n_pass / total
    print("WRO_W8A16_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    a, qw, sc, zr, g = build_inputs(M, K, N, GROUP_SIZE, seed=0)
    for _ in range(WARMUP):
        w8a16_matmul(a, qw, sc, zr, g)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        w8a16_matmul(a, qw, sc, zr, g)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS, "m": M, "n": N, "k": K,
           "group_size": GROUP_SIZE}
    # SOL: small-M weight-heavy dequant matmul. The 1-byte int8 weight stream
    # dominates traffic; the naive baseline additionally materialises the full fp32 weight
    # plus two expanded fp32 [K,N] scale/zero grids -> deeply memory-bound.
    flops = 2.0 * M * K * N
    G = K // GROUP_SIZE
    bytes_moved = (M * K * 2) + (K * N * 1) + (G * N * 2) + (G * N * 1) + (M * N * 2)
    if _HAVE_H20:
        try:
            peaks = load_peaks()
            r = roofline_t_sol(flops=flops, bytes_moved=bytes_moved, dtype="fp16", peaks=peaks)
            frac = sol_fraction(ms / 1e3, flops=flops, bytes_moved=bytes_moved,
                                dtype="fp16", peaks=peaks)
            res.update({"flops": flops, "bytes_moved": bytes_moved,
                        "t_sol_ms": round(r["t_sol_s"] * 1e3, 6), "bound": r["bound"],
                        "sol_fraction": round(frac, 6), "peaks_origin": r["peaks_origin"]})
        except Exception as e:
            res["sol_error"] = str(e)[:80]
    print("WRO_W8A16_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_W8A16_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_W8A16_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
