#!/usr/bin/env python3
"""Standalone workload for the 2:4 semi-structured sparse weight / fp16-activation
matmul subsystem (``sp24mm.sp24mm_matmul``).

Drives the PUBLIC entry ``sp24mm_matmul(a, w_vals, w_meta)`` (imported from the baked
/app/repo tree) with synthetic fixed-seed tensors on the GPU. The logical weight ``W``
is ``[K, N]`` fp16, 2:4 semi-structured sparse along K (exactly 2 nonzeros per 4-row
group), stored COMPRESSED as ``w_vals`` ``[K//2, N]`` (the 2 nonzero values per group,
K-order) + ``w_meta`` ``[K//4, N]`` uint8 (the two 2-bit in-group nonzero indices per
group). The subsystem returns ``a @ W`` with an fp32 accumulator. The benchmark drives a
small-M, large-K/N (weight-heavy) shape: the compressed weight is half the dense size.

Two modes:
  correctness : run the subsystem over a DIVERSE hidden suite of (M, K, N) and compare
                each against an INDEPENDENT fp32 reference (dense reconstruction here,
                NOT part of the editable scope), by relative-norm tolerance. Emits a
                pass-FRACTION over the whole suite (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on the big
                weight-heavy regime; also reports sol_fraction against the H20 roofline.

Emits one line ``WRO_SP24MM_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from sp24mm import sp24mm_matmul

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
DTYPE = torch.float16
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
WARMUP = 10
ITERS = 30

# ---- hidden correctness suite: many diverse (M, K, N) shapes (K a multiple of 4) ----
# The kernel must decode the per-(group,column) metadata (two 2-bit indices), place each
# of the 2 nonzeros at its correct in-group K row, and treat the other 2 rows as zero. A
# kernel that ignores the metadata, uses a fixed sparsity pattern, or only handles the
# timed shape fails these.
CORR_CASES = [
    (32, 512, 512),       # small
    (48, 768, 1024),      # non-power-of-2 M
    (64, 1024, 1024),     # square-ish
    (128, 2048, 512),     # tall K, skinny-ish N
    (96, 512, 4096),      # wide N
    (64, 2048, 320),      # skewed N
    (33, 1024, 4096),     # ragged M
    (256, 4096, 128),     # skinny N
    (16, 4096, 4096),     # large
    (64, 1536, 1536),     # mid
    (64, 1024, 768),      # rectangular
    (80, 640, 640),       # non-tile-mult M
    (64, 3072, 1024),     # tall K
    (32, 896, 896),       # prime-ish dims
]


def build_inputs(m, k, n, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, dtype=DTYPE, generator=g) * 0.05
    Kg = k // 4
    # dense 2:4 sparse weight: exactly 2 nonzeros per 4-row K-group, per column
    dense = torch.zeros((k, n), device=device, dtype=torch.float32)
    vals = torch.randn((Kg, 2, n), device=device, dtype=torch.float32, generator=g) * 0.1
    # pick 2 distinct in-group rows per (group, column) in ascending order
    # random permutation of [0,1,2,3] per (group,col), take first 2, sort ascending
    rnd = torch.rand((Kg, 4, n), device=device, generator=g)
    order = rnd.argsort(dim=1)          # [Kg,4,n] permutation of rows
    picks = order[:, :2, :]             # [Kg,2,n] the two chosen rows
    picks, _ = picks.sort(dim=1)        # ascending
    i0 = picks[:, 0, :]                 # [Kg,n]
    i1 = picks[:, 1, :]
    # scatter values into the dense group weight
    grp = torch.zeros((Kg, 4, n), device=device, dtype=torch.float32)
    nidx = torch.arange(n, device=device)[None, :].expand(Kg, n)
    gidx = torch.arange(Kg, device=device)[:, None].expand(Kg, n)
    grp[gidx, i0, nidx] = vals[:, 0, :]
    grp[gidx, i1, nidx] = vals[:, 1, :]
    dense = grp.reshape(k, n)
    # compress: w_vals [K//2,N] = the 2 nonzero values K-order; w_meta [K//4,N] = 2-bit indices
    w_vals = torch.empty((k // 2, n), device=device, dtype=DTYPE)
    w_vals[0::2, :] = vals[:, 0, :].to(DTYPE)
    w_vals[1::2, :] = vals[:, 1, :].to(DTYPE)
    w_meta = (i0 | (i1 << 2)).to(torch.uint8)     # [Kg, n]
    return a, w_vals, w_meta, dense


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def _ref(a, dense):
    return torch.matmul(a.to(torch.float32), dense)


def correctness():
    torch.cuda.synchronize()
    n_pass = 0
    detail = {}
    for i, (m, k, n) in enumerate(CORR_CASES):
        a, wv, wm, dense = build_inputs(m, k, n, seed=i)
        key = f"{m}x{k}x{n}"
        try:
            o = sp24mm_matmul(a, wv, wm).float()
        except Exception as e:
            detail[key] = {"error": str(e)[:80], "passed": False}
            continue
        ref = _ref(a, dense)
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
    print("WRO_SP24MM_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    a, wv, wm, dense = build_inputs(M, K, N, seed=0)
    for _ in range(WARMUP):
        sp24mm_matmul(a, wv, wm)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        sp24mm_matmul(a, wv, wm)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS, "m": M, "n": N, "k": K}
    # SOL: small-M weight-heavy 2:4 sparse matmul. The compressed weight
    # (w_vals K/2*N fp16 + w_meta K/4*N uint8) is streamed; the naive baseline additionally
    # materialises a full dense [K,N] fp16 weight -> memory-bound. Useful work = a @ W (fp32).
    flops = 2.0 * M * K * N
    bytes_moved = (M * K * 2) + (K // 2 * N * 2) + (K // 4 * N * 1) + (M * N * 2)  # a + w_vals + w_meta + c
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
    print("WRO_SP24MM_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_SP24MM_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_SP24MM_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
