#!/usr/bin/env python3
"""Standalone workload for the native sparse attention SELECTION operator
(``fla.ops.nsa.parallel_nsa``).

Drives the PUBLIC entry ``parallel_nsa(q, k, v, block_indices, block_counts,
block_size)`` (imported from the baked /app/repo tree) with synthetic fixed-seed
tensors on the GPU. Scored regime: the compressed / sliding-window branches are OFF
(``g_cmp = g_slc = g_swa = None``, ``window_size = 0``) and the per-query selected
block indices are provided directly, so the subsystem reduces to grouped-query
SELECTION attention -- each query attends, under a causal mask, only to the key/value
tokens inside its selected blocks.

Two modes:

  correctness : run the subsystem over a DIVERSE hidden suite of shapes (varying
                context length T incl. non-power-of-two multi-block lengths, varying
                batch B, varying number of selected blocks S, varying head width K/V)
                and compare each against an INDEPENDENT fp32 selection-attention
                reference computed here (NOT part of the editable scope), by
                relative-norm tolerance. Emits a pass FRACTION over the whole suite
                (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on a long-context
                regime; also reports sol_fraction against the H20 roofline. The
                degraded form runs an eager per-query loop over the selected blocks,
                so its wall time grows with the context length T while the useful work
                per query stays bounded at S*block_size keys -> the gap (and distance
                below the roofline) GROWS with T.

Emits one line ``WRO_NSA_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from fla.ops.nsa import parallel_nsa

# SOL helper (pure-math paths are import-safe anywhere; ship alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

# ---- timed regime: long context (per-query eager loop; headroom grows with T) ----
B_T = 1
T_T = 4096          # TQ == TK
H_T = 1             # kv heads
G_T = 16            # GQA group size (query heads per kv head); NSA requires G>=16, pow2
HQ_T = H_T * G_T    # query heads = 16
K_T = 128           # head dim (NSA forward requires K<=256, one K tile)
V_T = 128
BS_T = 64           # selected block size
S_T = 16            # selected blocks per query (block_counts); sparse: 16 of up to 64
DTYPE = torch.bfloat16
# bf16 GPU-kernel parity is judged by RELATIVE-NORM (not elementwise allclose, too
# strict given bf16 rounding between a blocked online-softmax kernel and an fp32
# gather-softmax reference): relative max-abs + L2.
REL_MAX_TOL = 3e-2
REL_L2_TOL = 1.5e-2
WARMUP = 3
ITERS = 10

# ---- hidden correctness suite: many diverse shapes (discrimination) ----
# (B, T, K, V, S). T mixes power-of-two and NON-power-of-two multi-block context
# lengths (all multiples of the 64-token selection block); a kernel that hard-codes the
# timed length, or mishandles the causal mask across a varying number of blocks, fails
# these. S (selected-block count) varies the sparsity -> exercises the padding (-1)
# entries and the per-query block loop bound. B varies -> exercises batch independence.
# K/V vary over the two head widths {64,128}. H=1 kv head with G=16 query heads holds
# (NSA requires G>=16 pow2). A naive impl correct only for the single timed shape
# scores a low fraction.
CORR_SHAPES = [
    (1, 512, 128, 128, 16),   (2, 512, 64, 64, 8),
    (1, 768, 128, 128, 12),   (2, 1024, 64, 128, 16),
    (1, 1024, 128, 128, 8),   (1, 1280, 64, 64, 16),
    (2, 640, 128, 128, 10),   (1, 1536, 128, 128, 16),
    (1, 896, 64, 64, 14),     (2, 448, 128, 64, 7),
    (1, 2048, 128, 128, 16),  (1, 1792, 64, 128, 12),
]
BLOCK_SIZE = 64
G = 16
H = 1


def _build_block_indices(B, T, S, device):
    """Deterministic causal 'sink + recent-window' block selection, shape [B,T,H,S].

    For a query whose current block is ib (= t // BS):
      * if ib+1 <= S : select all causally-valid blocks 0..ib (rest padded -1);
      * else         : select block 0 (sink) + the last S-1 blocks (ib-S+2 .. ib).
    All selected blocks are causally valid (block start <= t) and distinct.
    """
    nb = T // BLOCK_SIZE
    tbl = torch.full((T, S), -1, dtype=torch.long)
    for t in range(T):
        ib = t // BLOCK_SIZE
        if ib + 1 <= S:
            picks = list(range(ib + 1))
        else:
            picks = [0] + list(range(ib - S + 2, ib + 1))
        for i, b in enumerate(picks):
            tbl[t, i] = b
    return tbl.view(1, T, 1, S).expand(B, T, H, S).contiguous().to(device)


def build_inputs(B, T, K, V, S, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=gen)
    HQ = H * G
    q = r(B, T, HQ, K)
    k = r(B, T, H, K)
    v = r(B, T, H, V)
    block_indices = _build_block_indices(B, T, S, device)
    return dict(q=q, k=k, v=v, block_indices=block_indices,
                block_counts=S, block_size=BLOCK_SIZE)


def run_scope(inp):
    o = parallel_nsa(
        q=inp["q"], k=inp["k"], v=inp["v"],
        block_indices=inp["block_indices"],
        block_counts=inp["block_counts"],
        block_size=inp["block_size"],
    )
    if isinstance(o, tuple):
        o = o[0]
    return o


def nsa_selection_reference(inp):
    """Independent trusted fp32 selection-attention reference (ground truth; NOT in
    the editable scope). For each query, softmax over the causally-visible key tokens
    that lie in the query's selected blocks (scale = K**-0.5), GQA-aware, chunked over
    query positions to bound memory."""
    q = inp["q"]; k = inp["k"]; v = inp["v"]; bidx = inp["block_indices"]
    BS = inp["block_size"]
    B, TQ, HQd, Kd = q.shape
    TK, Hd, Vd = k.shape[1], k.shape[2], v.shape[-1]
    Gd = HQd // Hd
    Sd = bidx.shape[-1]
    N = Sd * BS
    scale = Kd ** -0.5
    kbh = k.float().permute(0, 2, 1, 3)     # [B,H,TK,K]
    vbh = v.float().permute(0, 2, 1, 3)     # [B,H,TK,V]
    o = q.new_zeros(B, TQ, HQd, Vd, dtype=torch.float32)
    ar = torch.arange(BS, device=q.device)
    BQ = 256
    for s0 in range(0, TQ, BQ):
        s1 = min(s0 + BQ, TQ)
        bq = s1 - s0
        bi = bidx[:, s0:s1]                                  # [B,bq,H,S]
        tok = (bi.unsqueeze(-1) * BS + ar).reshape(B, bq, Hd, N)   # [B,bq,H,N]
        valid = tok >= 0
        tokc = tok.clamp(0, TK - 1)
        idxk = tokc.permute(0, 2, 1, 3)                      # [B,H,bq,N]
        kg = torch.gather(kbh.unsqueeze(2).expand(B, Hd, bq, TK, Kd), 3,
                          idxk.unsqueeze(-1).expand(B, Hd, bq, N, Kd))   # [B,H,bq,N,K]
        vg = torch.gather(vbh.unsqueeze(2).expand(B, Hd, bq, TK, Vd), 3,
                          idxk.unsqueeze(-1).expand(B, Hd, bq, N, Vd))   # [B,H,bq,N,V]
        qg = (q[:, s0:s1].float() * scale).reshape(B, bq, Hd, Gd, Kd).permute(0, 2, 3, 1, 4)  # [B,H,G,bq,K]
        sc = torch.einsum('bhgqk,bhqnk->bhgqn', qg, kg)      # [B,H,G,bq,N]
        qpos = torch.arange(s0, s1, device=q.device).view(1, bq, 1, 1)
        bad = (tok > qpos) | (~valid)                        # [B,bq,H,N]
        bad = bad.permute(0, 2, 1, 3).unsqueeze(2)           # [B,H,1,bq,N]
        sc = sc.masked_fill(bad, float('-inf'))
        w = torch.nan_to_num(torch.softmax(sc, dim=-1), nan=0.0)
        ob = torch.einsum('bhgqn,bhqnv->bhgqv', w, vg)       # [B,H,G,bq,V]
        ob = ob.permute(0, 3, 1, 2, 4).reshape(B, bq, HQd, Vd)
        o[:, s0:s1] = ob
    return o


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def correctness():
    torch.cuda.synchronize()
    n_pass = 0
    detail = {}
    for i, (B, T, K, V, S) in enumerate(CORR_SHAPES):
        inp = build_inputs(B, T, K, V, S, seed=i)
        tag = f"{B}x{T}x{K}x{V}xS{S}"
        try:
            o = run_scope(inp).float()
        except Exception as e:
            detail[tag] = {"error": type(e).__name__ + ":" + str(e)[:70], "passed": False}
            continue
        ref = nsa_selection_reference(inp)
        torch.cuda.synchronize()
        HQ = H * G
        if list(o.shape) != [B, T, HQ, V]:
            detail[tag] = {"shape": list(o.shape), "passed": False}
            continue
        rel_max, rel_l2, _ = _relnorm(o, ref)
        ok = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        n_pass += int(ok)
        detail[tag] = {"rel_max": round(rel_max, 5), "rel_l2": round(rel_l2, 5), "passed": bool(ok)}
    total = len(CORR_SHAPES)
    frac = n_pass / total
    print("WRO_NSA_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    inp = build_inputs(B_T, T_T, K_T, V_T, S_T, seed=0)
    for _ in range(WARMUP):
        run_scope(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run_scope(inp)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS,
           "batch": B_T, "seqlen": T_T, "hq": HQ_T, "h": H_T, "k": K_T, "v": V_T,
           "block_size": BS_T, "s": S_T}
    # SOL. The scored selection attention does, per query and query head,
    # a QK^T and a PV over the S*block_size selected key tokens:
    #   FLOPs ~ 2 * B*HQ*T * (S*BS) * (K + V)   (QK contraction + PV contraction)
    # The compute-optimal blocked kernel streams the selected K/V blocks with an online
    # softmax; the roofline ceiling for that arithmetic is the SOL anchor. The degraded
    # eager per-query loop is launch/bandwidth bound and sits far below it.
    n_ctx = S_T * BS_T
    flops = 2.0 * B_T * HQ_T * T_T * n_ctx * (K_T + V_T)
    # bytes: read q [B,T,HQ,K] + gather S*BS keys/values per query (bounded reuse) + write o
    q_bytes = B_T * T_T * HQ_T * K_T
    kv_bytes = B_T * H_T * T_T * (K_T + V_T)   # each kv token read O(1) times on average
    o_bytes = B_T * T_T * HQ_T * V_T
    bytes_moved = (q_bytes + kv_bytes + o_bytes) * 2  # bf16 lower bound
    if _HAVE_H20:
        try:
            peaks = load_peaks()
            r = roofline_t_sol(flops=flops, bytes_moved=bytes_moved, dtype="bf16", peaks=peaks)
            frac = sol_fraction(ms / 1e3, flops=flops, bytes_moved=bytes_moved,
                                dtype="bf16", peaks=peaks)
            res.update({"flops": flops, "bytes_moved": bytes_moved,
                        "t_sol_ms": round(r["t_sol_s"] * 1e3, 6), "bound": r["bound"],
                        "sol_fraction": round(frac, 6), "peaks_origin": r["peaks_origin"]})
        except Exception as e:
            res["sol_error"] = str(e)[:80]
    print("WRO_NSA_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_NSA_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_NSA_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
