#!/usr/bin/env python3
"""Standalone workload for the BLOCK-SPARSE multi-head attention FORWARD subsystem
(flash_attn.cute).

Drives the in-scope public entry ``flash_attn.cute.flash_attn_func`` imported from the
baked /app/repo flash-attention tree, exercising a workload-defined block-band
``BlockSparseTensorsTorch`` (block partitioning of Q along the query axis and K/V along
the key axis; each query block m is allowed to attend to key blocks in a bounded band
[m - radius, m + radius], with all included block-pairs "fully covered" — no per-position
masking inside a block). The fused CuTe kernel walks the ordered integer index arrays
and skips whole (m_block, kv_block) pairs at scheduler level via
block_sparse_utils.produce_block_sparse_loads.

Correctness is graded as a PASS-RATE over MANY diverse hidden cases against an
INDEPENDENT fp32 reference that materializes the same block-band bool mask on the
score matrix. Timing runs one forward over a long-sequence sparse workload. Emits
WRO_FA_RESULT.

Usage: python3 workload.py {correctness|timing}
"""
import json
import math
import sys
import time

import torch

from flash_attn.cute import flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


REL_L2_TOL = 3e-2
REL_MAX_TOL = 8e-2
WARMUP = 6
ITERS = 20

# Long-sequence memory-bound timing: 8k x 8k, block=128, band radius=2 -> ~5/64 = 7.8%
# density -> fused skips >90% of the (m_block, kv_block) pairs -> big HBM savings.
TIMING = dict(b=1, sq=8192, sk=8192, hq=8, hk=8, d=128,
              block_q=128, block_kv=128, radius=2, dtype=torch.bfloat16)


# A diverse hidden correctness suite:
#   (b, sq, sk, hq, hk, d, block_q, block_kv, radius, dtype, scale_mode)
# block_q must be a multiple of the Hopper CuTe fwd's q_stage * m_block_size (validated
# by normalize_block_sparse_config); block_kv must equal the kernel's tile_n. For
# headdim in {64, 128} on H20 sm90, both block_q=block_kv=128 satisfies the constraint.
# radius: number of KV-blocks on each side of the diagonal M-block that are included.
CORR_CASES = [
    (1, 512, 512, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (2, 512, 512, 8, 8, 128, 128, 128, 2, torch.bfloat16, "default"),
    (1, 768, 768, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (2, 768, 768, 8, 8, 128, 128, 128, 2, torch.bfloat16, "custom"),
    (1, 1024, 1024, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (1, 1024, 1024, 8, 2, 128, 128, 128, 2, torch.bfloat16, "default"),   # GQA 4x
    (2, 1024, 1024, 16, 4, 128, 128, 128, 1, torch.bfloat16, "default"),  # GQA 4x
    (1, 1536, 1536, 8, 8, 128, 128, 128, 3, torch.bfloat16, "default"),
    (2, 1536, 1536, 8, 2, 128, 128, 128, 2, torch.bfloat16, "custom"),   # GQA 4x
    (1, 2048, 2048, 8, 8, 128, 128, 128, 3, torch.bfloat16, "default"),
    (1, 2048, 2048, 16, 16, 128, 128, 128, 4, torch.bfloat16, "custom"),
    (2, 384, 384, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (1, 640, 640, 8, 8, 128, 128, 128, 2, torch.bfloat16, "default"),
    (2, 384, 640, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),    # cross-attn sq<sk
    (1, 640, 384, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),    # cross-attn sq>sk
    (1, 1024, 1024, 8, 8, 128, 128, 128, 2, torch.float16, "default"),  # fp16
    (2, 512, 512, 32, 32, 128, 128, 128, 1, torch.bfloat16, "default"),  # many heads
    (1, 4096, 4096, 8, 8, 128, 128, 128, 4, torch.bfloat16, "default"), # long
]


def _scale(d, mode):
    return (1.0 / math.sqrt(d)) if mode == "default" else (0.7 / math.sqrt(d))


def build(b, sq, sk, hq, hk, d, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    return q, k, v


def make_band_bst_and_dense(sq, sk, block_q, block_kv, radius, device):
    """Return BlockSparseTensorsTorch (band pattern, all blocks in FULL slot) plus a
    dense (sq, sk) bool allow-mask for the fp32 reference."""
    num_m = (sq + block_q - 1) // block_q
    num_n = (sk + block_kv - 1) // block_kv
    rows = []
    for m in range(num_m):
        lo = max(0, m - radius); hi = min(num_n - 1, m + radius)
        rows.append(list(range(lo, hi + 1)))
    max_cnt = max(1, max(len(r) for r in rows))
    full_cnt = torch.tensor([len(r) for r in rows], dtype=torch.int32,
                            device=device).view(1, 1, num_m)
    full_idx = torch.zeros(1, 1, num_m, max_cnt, dtype=torch.int32, device=device)
    for i, r in enumerate(rows):
        for k_i, n_ in enumerate(r):
            full_idx[0, 0, i, k_i] = n_
    mask_cnt = torch.zeros(1, 1, num_m, dtype=torch.int32, device=device)
    mask_idx = torch.zeros(1, 1, num_m, 1, dtype=torch.int32, device=device)
    bst = BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(block_q, block_kv),
    )
    dense = torch.zeros(sq, sk, dtype=torch.bool, device=device)
    for m, r in enumerate(rows):
        for n_ in r:
            r0, r1 = m * block_q, min((m + 1) * block_q, sq)
            c0, c1 = n_ * block_kv, min((n_ + 1) * block_kv, sk)
            dense[r0:r1, c0:c1] = True
    return bst, dense


def reference(q, k, v, scale, dense_mask):
    """Independent fp32 reference: explicit materialized-score attention with the same
    block-band dense allow-mask. Handles GQA by repeating K/V heads."""
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    kk, vv = k, v
    if hk != hq:
        rep = hq // hk
        kk = k.repeat_interleave(rep, dim=2)
        vv = v.repeat_interleave(rep, dim=2)
    qf = q.transpose(1, 2).float()
    kf = kk.transpose(1, 2).float()
    vf = vv.transpose(1, 2).float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    m = dense_mask.view(1, 1, sq, sk).expand(b, hq, sq, sk)
    scores = scores.masked_fill(~m, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    return torch.matmul(p, vf).transpose(1, 2)


def rel_norms(cand, ref):
    cand = cand.to(torch.float32)
    ref = ref.to(torch.float32)
    denom = ref.norm().item() + 1e-9
    return ((cand - ref).norm().item() / denom,
            (cand - ref).abs().max().item() / (ref.abs().max().item() + 1e-9))


def run_scope(q, k, v, scale, bst):
    o = flash_attn_func(q, k, v, causal=False, softmax_scale=scale,
                       block_sparse_tensors=bst)
    return o[0] if isinstance(o, tuple) else o


def correctness():
    torch.cuda.init()
    n_pass = 0
    detail = {}
    for i, (b, sq, sk, hq, hk, d, bq, bkv, radius, dtype, sm) in enumerate(CORR_CASES):
        try:
            scale = _scale(d, sm)
            q, k, v = build(b, sq, sk, hq, hk, d, dtype, seed=100 + i)
            bst, dense = make_band_bst_and_dense(sq, sk, bq, bkv, radius, q.device)
            out = run_scope(q, k, v, scale, bst)
            ref = reference(q, k, v, scale, dense)
            l2, mx = rel_norms(out, ref)
            ok = (l2 <= REL_L2_TOL and mx <= REL_MAX_TOL)
            n_pass += int(ok)
            detail[f"case{i}"] = {"rel_l2": round(l2, 5), "rel_max": round(mx, 5), "passed": ok}
        except Exception as e:
            detail[f"case{i}"] = {"error": f"{type(e).__name__}: {str(e)[:160]}", "passed": False}
    total = len(CORR_CASES)
    frac = n_pass / total
    print("WRO_FA_RESULT " + json.dumps(
        {"correctness_ok": (n_pass == total), "correctness_frac": round(frac, 4),
         "n_pass": n_pass, "n_total": total, "detail": detail}))


def timing():
    torch.cuda.init()
    t = TIMING
    q, k, v = build(t["b"], t["sq"], t["sk"], t["hq"], t["hk"], t["d"], t["dtype"], seed=7)
    scale = 1.0 / math.sqrt(t["d"])
    bst, _ = make_band_bst_and_dense(t["sq"], t["sk"], t["block_q"], t["block_kv"],
                                     t["radius"], q.device)
    fn = lambda: run_scope(q, k, v, scale, bst)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / ITERS
    print("WRO_FA_RESULT " + json.dumps({"timing_ms": round(ms, 6)}))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_FA_RESULT " + json.dumps({"error": f"unknown mode {mode}"}))
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
