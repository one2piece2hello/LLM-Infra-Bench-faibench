#!/usr/bin/env python3
"""Standalone workload for the structured K-block-sparse weight / fp16-activation matmul
subsystem (``blocksp.blocksp_matmul``).

Drives the PUBLIC entry ``blocksp_matmul(a, w_blocks, k_idx, block_k)`` (imported from the
baked /app/repo tree) with synthetic fixed-seed tensors on the GPU. The logical weight
``W`` is ``[K, N]`` fp16, structured block-sparse along K (only ``nnz`` of ``K//block_k``
row-blocks nonzero), stored COMPRESSED as ``w_blocks`` ``[nnz*block_k, N]`` (nonzero blocks
stacked, ascending) + ``k_idx`` ``[nnz]`` int32 (logical block index of each). The
subsystem returns ``a @ W`` with an fp32 accumulator. The benchmark drives a small-M,
large-K/N weight-heavy shape at LOW block-density.

Two modes:
  correctness : run the subsystem over a DIVERSE hidden suite of (M, K, N, block_k, nnz)
                and compare each against an INDEPENDENT fp32 reference (dense
                reconstruction here, NOT part of the editable scope), by relative-norm
                tolerance. Emits a pass-FRACTION over the whole suite (graded
                correctness).
  timing      : warmup + timed repeats of the subsystem call only, on the big
                weight-heavy low-density regime; also reports sol_fraction vs the H20 roofline.

Emits one line ``WRO_BLOCKSP_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from blocksp import blocksp_matmul

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

# ---- timed regime: small-M, large-K/N, LOW block-density (weight-heavy, memory/compute-bound) ----
M = 64
K = 8192
N = 8192
BLOCK_K = 128
NNZ = 16                 # of K//BLOCK_K = 64 blocks -> 25% density
DTYPE = torch.float16
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
WARMUP = 10
ITERS = 30

# ---- hidden correctness suite: many diverse (M, K, N, block_k, nnz) shapes ----
# block_k is a power of two dividing K; nnz in [1, K//block_k]. The kernel must place each
# stored block at its k_idx logical position and treat all other blocks as zero. A kernel
# that assumes contiguous/all blocks, ignores k_idx, or only handles the timed shape fails.
# (M, K, N, block_k, nnz)
CORR_CASES = [
    (32, 1024, 512, 128, 3),      # small, sparse
    (48, 2048, 1024, 64, 10),     # non-pow2 M
    (64, 4096, 1024, 128, 8),     # square-ish
    (128, 2048, 512, 256, 4),     # big block
    (96, 1024, 4096, 64, 5),      # wide N
    (64, 4096, 320, 128, 20),     # skewed N
    (33, 2048, 4097, 128, 7),     # ragged M+N
    (256, 4096, 128, 128, 12),    # skinny N
    (16, 8192, 4096, 256, 6),     # large
    (64, 2048, 1536, 32, 40),     # tiny block, many nz
    (64, 1024, 768, 512, 1),      # one block only
    (80, 1280, 640, 128, 9),      # non-tile-mult M
    (64, 6144, 1024, 128, 3),     # tall K, very sparse
    (32, 1024, 896, 64, 16),      # all blocks nonzero (dense edge)
]


def build_inputs(m, k, n, block_k, nnz, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, dtype=DTYPE, generator=g) * 0.05
    num_blocks = k // block_k
    nnz = min(nnz, num_blocks)
    # choose nnz distinct block indices (ascending)
    perm = torch.randperm(num_blocks, generator=g, device=device)
    k_idx = perm[:nnz].sort().values.to(torch.int32)
    w_blocks = torch.randn((nnz * block_k, n), device=device, dtype=DTYPE, generator=g) * 0.1
    # dense reference weight
    dense = torch.zeros((k, n), device=device, dtype=torch.float32)
    for p in range(nnz):
        kb = int(k_idx[p].item())
        dense[kb * block_k:(kb + 1) * block_k, :] = w_blocks[p * block_k:(p + 1) * block_k, :].to(torch.float32)
    return a, w_blocks, k_idx, block_k, dense


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
    for i, (m, k, n, bk, nz) in enumerate(CORR_CASES):
        a, wb, kidx, block_k, dense = build_inputs(m, k, n, bk, nz, seed=i)
        key = f"{m}x{k}x{n}_bk{bk}_nnz{kidx.shape[0]}"
        try:
            o = blocksp_matmul(a, wb, kidx, block_k).float()
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
    print("WRO_BLOCKSP_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    a, wb, kidx, block_k, dense = build_inputs(M, K, N, BLOCK_K, NNZ, seed=0)
    for _ in range(WARMUP):
        blocksp_matmul(a, wb, kidx, block_k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        blocksp_matmul(a, wb, kidx, block_k)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS, "m": M, "n": N, "k": K,
           "block_k": BLOCK_K, "nnz": NNZ, "num_blocks": K // BLOCK_K}
    # SOL: only the nnz stored blocks contribute. Useful work = a[:, active] @
    # w_blocks (2*M*(nnz*block_k)*N flops); useful traffic = a active cols + w_blocks + c.
    # The naive densifies to [K,N] and does the full dense matmul -> memory+compute bound.
    keff = NNZ * BLOCK_K
    flops = 2.0 * M * keff * N
    bytes_moved = (M * keff * 2) + (keff * N * 2) + (M * N * 2)  # active a + stored w + c
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
    print("WRO_BLOCKSP_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_BLOCKSP_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_BLOCKSP_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
