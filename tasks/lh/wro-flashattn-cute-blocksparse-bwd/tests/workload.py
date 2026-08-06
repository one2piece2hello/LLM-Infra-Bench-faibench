#!/usr/bin/env python3
"""Standalone workload for the BLOCK-SPARSE multi-head attention BACKWARD subsystem
(flash_attn.cute).

Drives the in-scope public entry ``flash_attn.cute.flash_attn_func`` imported from the baked
/app/repo flash-attention tree in a TRAINING step with block-sparse restrictions: it runs the
forward, then backpropagates a random upstream gradient and reads the input gradients dq, dk, dv.
The workload constructs a band-partition ``BlockSparseTensorsTorch`` (each query M-block attends
to a small set of KV-blocks within a bounded band around the diagonal, all block-pairs "fully
covered" — no per-position masking inside a block) plus the corresponding backward Q-direction
transpose plus the ``dq_write_order`` semaphore metadata (compute_dq_write_order), i.e. exactly
the arguments the public entry documents.

Correctness is graded as a PASS-RATE over MANY diverse hidden cases (varied batch, sequence
lengths, head counts incl. grouped-query, head_dim 128, dtype, softmax scale, and block-band
radii): for each case the output AND all three input gradients dq, dk, dv must match an
INDEPENDENT fp32 autograd reference (gradients of an explicit materialized band-masked-score
attention) within tolerance. Timing measures one forward+backward step over a long-sequence
block-sparse workload. Emits WRO_FA_RESULT.

Usage: python3 workload.py {correctness|timing}
"""
import json
import math
import sys
import time

import torch

from flash_attn.cute import flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch, compute_dq_write_order


REL_L2_TOL = 3e-2
REL_MAX_TOL = 8e-2
WARMUP = 5
ITERS = 15

# Long-sequence fwd+bwd timing case: 4k x 4k, block=128, band radius=2
# -> ~5/32 = 15.6% of block pairs are inside the band.
TIMING = dict(b=1, sq=4096, sk=4096, hq=8, hk=8, d=128,
              block_q=128, block_kv=128, radius=2, dtype=torch.bfloat16)


# A diverse hidden correctness suite:
#   (b, sq, sk, hq, hk, d, block_q, block_kv, radius, dtype, scale_mode)
# All non-causal; band-partition (each M-block attends to N-blocks within [m-radius, m+radius]).
# For H20 sm90 block-sparse bwd, block_q must equal block_kv and both must be 128 (the fused
# _flash_attn_bwd tile). radius chosen strictly < num_m_blocks so the pattern genuinely bites.
CORR_CASES = [
    (2, 512, 512, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (2, 512, 512, 8, 8, 128, 128, 128, 2, torch.bfloat16, "default"),
    (1, 768, 768, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (2, 768, 768, 8, 8, 128, 128, 128, 2, torch.bfloat16, "custom"),
    (1, 1024, 1024, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (1, 1024, 1024, 8, 2, 128, 128, 128, 2, torch.bfloat16, "default"),  # GQA 4x
    (2, 1024, 1024, 16, 4, 128, 128, 128, 1, torch.bfloat16, "default"),  # GQA 4x
    (1, 1536, 1536, 8, 8, 128, 128, 128, 3, torch.bfloat16, "default"),
    (2, 1536, 1536, 8, 2, 128, 128, 128, 2, torch.bfloat16, "custom"),  # GQA 4x
    (1, 2048, 2048, 8, 8, 128, 128, 128, 3, torch.bfloat16, "default"),
    (1, 2048, 2048, 16, 16, 128, 128, 128, 4, torch.bfloat16, "custom"),
    (2, 384, 384, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),
    (1, 640, 640, 8, 8, 128, 128, 128, 2, torch.bfloat16, "default"),
    (2, 384, 640, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),  # cross-attn sq<sk
    (1, 640, 384, 8, 8, 128, 128, 128, 1, torch.bfloat16, "default"),  # cross-attn sq>sk
    (1, 1024, 1024, 8, 8, 128, 128, 128, 2, torch.float16, "default"),  # fp16
    (2, 512, 512, 32, 32, 128, 128, 128, 1, torch.bfloat16, "default"),  # many heads
    (1, 3072, 3072, 8, 8, 128, 128, 128, 4, torch.bfloat16, "default"),  # long
]


def _scale(d, mode):
    return (1.0 / math.sqrt(d)) if mode == "default" else (0.7 / math.sqrt(d))


def build(b, sq, sk, hq, hk, d, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    return q, k, v


def _build_bst_fwd(sq, sk, block_q, block_kv, radius, device):
    """Build the forward BlockSparseTensorsTorch (KV-direction ordered index arrays):
    for each query M-block m, list the allowed KV-blocks (band radius around m). All blocks
    in the FULL slot (fully-covered blocks — no per-position mask needed inside a block); the
    MASK slot carries an all-zeros count with a valid dummy index array shape (see SA-Ck6's
    trap 6). Broadcast (batch=1, head=1) across batch and head."""
    num_m = (sq + block_q - 1) // block_q
    num_n = (sk + block_kv - 1) // block_kv
    rows_kv = []
    for m in range(num_m):
        lo = max(0, m - radius); hi = min(num_n - 1, m + radius)
        rows_kv.append(list(range(lo, hi + 1)))
    max_cnt = max(1, max(len(r) for r in rows_kv))
    full_cnt = torch.tensor([len(r) for r in rows_kv], dtype=torch.int32,
                            device=device).view(1, 1, num_m)
    full_idx = torch.zeros(1, 1, num_m, max_cnt, dtype=torch.int32, device=device)
    for i, r in enumerate(rows_kv):
        for k_i, n_ in enumerate(r):
            full_idx[0, 0, i, k_i] = n_
    mask_cnt = torch.zeros(1, 1, num_m, dtype=torch.int32, device=device)
    mask_idx = torch.zeros(1, 1, num_m, 1, dtype=torch.int32, device=device)
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(block_q, block_kv),
    ), rows_kv


def _build_bst_bwd(sq, sk, block_q, block_kv, rows_kv, device):
    """Build the backward Q-direction BlockSparseTensorsTorch by transposing the forward band
    pattern: for each key N-block n, list the query M-blocks whose fwd rows include n. Compute
    the dq_write_order semaphore metadata (rank of each n_block inside the target m_block's
    combined contributor list) via block_sparsity.compute_dq_write_order so the fused block-
    sparse backward CuTe kernel can accumulate dQ deterministically without atomics on shared
    tiles."""
    num_m = (sq + block_q - 1) // block_q
    num_n = (sk + block_kv - 1) // block_kv
    # Transpose: for each n_block, collect the list of m_blocks that reference it fwd.
    rows_q = [[] for _ in range(num_n)]
    for m, r in enumerate(rows_kv):
        for n_ in r:
            rows_q[n_].append(m)
    max_cnt_q = max(1, max(len(r) for r in rows_q))
    full_cnt_bwd = torch.tensor([len(r) for r in rows_q], dtype=torch.int32,
                                device=device).view(1, 1, num_n)
    full_idx_bwd = torch.zeros(1, 1, num_n, max_cnt_q, dtype=torch.int32, device=device)
    for j, r in enumerate(rows_q):
        for k_i, m in enumerate(r):
            full_idx_bwd[0, 0, j, k_i] = m
    mask_cnt_bwd = torch.zeros(1, 1, num_n, dtype=torch.int32, device=device)
    mask_idx_bwd = torch.zeros(1, 1, num_n, 1, dtype=torch.int32, device=device)
    # Now compute the write-order metadata (dq_write_order / dq_write_order_full).
    # Rebuild the forward index arrays for compute_dq_write_order.
    max_cnt_fwd = max(1, max(len(r) for r in rows_kv))
    fwd_full_cnt = torch.tensor([len(r) for r in rows_kv], dtype=torch.int32,
                                device=device).view(1, 1, num_m)
    fwd_full_idx = torch.zeros(1, 1, num_m, max_cnt_fwd, dtype=torch.int32, device=device)
    for i, r in enumerate(rows_kv):
        for k_i, n_ in enumerate(r):
            fwd_full_idx[0, 0, i, k_i] = n_
    fwd_mask_cnt = torch.zeros(1, 1, num_m, dtype=torch.int32, device=device)
    fwd_mask_idx = torch.zeros(1, 1, num_m, 1, dtype=torch.int32, device=device)
    dq_wo, dq_wo_full = compute_dq_write_order(
        fwd_mask_cnt, fwd_mask_idx, fwd_full_cnt, fwd_full_idx,
        mask_cnt_bwd, mask_idx_bwd, full_cnt_bwd, full_idx_bwd, spt=False,
    )
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt_bwd,
        mask_block_idx=mask_idx_bwd,
        full_block_cnt=full_cnt_bwd,
        full_block_idx=full_idx_bwd,
        dq_write_order=dq_wo,
        dq_write_order_full=dq_wo_full,
        block_size=(block_q, block_kv),
        spt=False,
    )


def _dense_allow(rows_kv, sq, sk, block_q, block_kv, device):
    """Rebuild the dense (sq, sk) bool allow-mask from the workload row-plan for the fp32
    reference. Independent implementation from the fused kernel's block-list-driven scheduler."""
    dense = torch.zeros(sq, sk, dtype=torch.bool, device=device)
    for m, r in enumerate(rows_kv):
        for n_ in r:
            r0, r1 = m * block_q, min((m + 1) * block_q, sq)
            c0, c1 = n_ * block_kv, min((n_ + 1) * block_kv, sk)
            dense[r0:r1, c0:c1] = True
    return dense


def reference_grads(q, k, v, scale, dense_mask, dout):
    """Independent fp32 autograd reference: gradients of an explicit materialized band-masked
    score attention. Grouped-query supported."""
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    qd = q.detach().float().requires_grad_(True)
    kd = k.detach().float().requires_grad_(True)
    vd = v.detach().float().requires_grad_(True)
    kk, vv = kd, vd
    if hk != hq:
        rep = hq // hk
        kk = kd.repeat_interleave(rep, dim=2)
        vv = vd.repeat_interleave(rep, dim=2)
    qf = qd.transpose(1, 2)
    kf = kk.transpose(1, 2)
    vf = vv.transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    m = dense_mask.view(1, 1, sq, sk).expand(b, hq, sq, sk)
    scores = scores.masked_fill(~m, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.matmul(p, vf).transpose(1, 2)
    dq, dk, dv = torch.autograd.grad(o, (qd, kd, vd), dout.float())
    return o, dq, dk, dv


def rel_norms(cand, ref):
    cand = cand.to(torch.float32)
    ref = ref.to(torch.float32)
    denom = ref.norm().item() + 1e-9
    return ((cand - ref).norm().item() / denom,
            (cand - ref).abs().max().item() / (ref.abs().max().item() + 1e-9))


def run_scope_fwd_bwd(q, k, v, scale, bst_fwd, bst_bwd, dout):
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    o = flash_attn_func(qg, kg, vg, causal=False, softmax_scale=scale,
                        block_sparse_tensors=bst_fwd,
                        block_sparse_tensors_bwd=bst_bwd)
    o = o[0] if isinstance(o, tuple) else o
    o.backward(dout)
    return o, qg.grad, kg.grad, vg.grad


def correctness():
    torch.cuda.init()
    n_pass = 0
    detail = {}
    for i, (b, sq, sk, hq, hk, d, bq, bkv, radius, dtype, sm) in enumerate(CORR_CASES):
        try:
            scale = _scale(d, sm)
            q, k, v = build(b, sq, sk, hq, hk, d, dtype, seed=100 + i)
            bst_fwd, rows_kv = _build_bst_fwd(sq, sk, bq, bkv, radius, q.device)
            bst_bwd = _build_bst_bwd(sq, sk, bq, bkv, rows_kv, q.device)
            dense = _dense_allow(rows_kv, sq, sk, bq, bkv, q.device)
            g = torch.Generator(device="cuda").manual_seed(700 + i)
            dout = torch.randn(b, sq, hq, d, device="cuda", dtype=torch.float32,
                               generator=g).to(dtype)
            o, dq, dk, dv = run_scope_fwd_bwd(q, k, v, scale, bst_fwd, bst_bwd, dout)
            ro, rdq, rdk, rdv = reference_grads(q, k, v, scale, dense, dout)
            lo, mo = rel_norms(o, ro)
            lq, mq = rel_norms(dq, rdq)
            lk, mk = rel_norms(dk, rdk)
            lv, mv = rel_norms(dv, rdv)
            ok = all(l <= REL_L2_TOL for l in (lo, lq, lk, lv)) and \
                 all(m <= REL_MAX_TOL for m in (mo, mq, mk, mv))
            n_pass += int(ok)
            detail[f"case{i}"] = {"o": round(lo, 5), "dq": round(lq, 5), "dk": round(lk, 5),
                                  "dv": round(lv, 5), "passed": ok}
        except Exception as e:
            detail[f"case{i}"] = {"error": f"{type(e).__name__}: {str(e)[:120]}", "passed": False}
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
    bst_fwd, rows_kv = _build_bst_fwd(t["sq"], t["sk"], t["block_q"], t["block_kv"],
                                      t["radius"], q.device)
    bst_bwd = _build_bst_bwd(t["sq"], t["sk"], t["block_q"], t["block_kv"], rows_kv, q.device)
    g = torch.Generator(device="cuda").manual_seed(7)
    dout = torch.randn(t["b"], t["sq"], t["hq"], t["d"], device="cuda", dtype=torch.float32,
                       generator=g).to(t["dtype"])

    def step():
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        o = flash_attn_func(qg, kg, vg, causal=False, softmax_scale=scale,
                            block_sparse_tensors=bst_fwd,
                            block_sparse_tensors_bwd=bst_bwd)
        o = o[0] if isinstance(o, tuple) else o
        o.backward(dout)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
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
