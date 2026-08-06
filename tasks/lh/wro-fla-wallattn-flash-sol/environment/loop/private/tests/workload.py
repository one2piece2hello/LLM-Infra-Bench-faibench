#!/usr/bin/env python3
"""Standalone workload for the windowed decay-attention subsystem
(``fla.ops.wall_attn.parallel_wall_attn``).

Drives the PUBLIC entry ``parallel_wall_attn(q, k, v, g, window_size=...)`` (imported
from the baked /app/repo tree) with synthetic fixed-seed tensors on the GPU. The op is
a causal, sliding-window attention with a per-channel multiplicative DECAY applied
through a log-space prefix sum: with ``P = cumsum(g) * 1/ln2`` (log2-space), the logit
for query ``i`` / key ``j`` is ``scale * (1/ln2) * sum_n q[i,n] k[j,n] exp2(P[i,n] -
P[j,n])`` under a causal + window mask, softmaxed over keys (in exp2 space), then
``o = weights @ v``.

Two modes:

  correctness : run the subsystem over a DIVERSE hidden suite of shapes (varying
                context length T incl. non-power-of-two, varying batch B, varying head
                count HQ and GQA grouping, varying sliding-window width, varying head
                width K/V) and compare each against an INDEPENDENT fp32 windowed-decay
                softmax reference computed here (NOT part of the editable scope, and
                NOT the shipped naive path), by relative-norm tolerance. Emits a pass
                FRACTION over the whole suite (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on a long-context
                regime; also reports sol_fraction against the H20 roofline. The degraded
                form materializes the full O(T^2) pairwise-score tensor and softmaxes
                densely (no streaming, no window skipping), so its work and memory grow
                with T^2 while the useful windowed work grows ~ T*W -> the gap and the
                distance below the roofline GROW with T.

Emits one line ``WRO_WALLATTN_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from fla.ops.wall_attn import parallel_wall_attn

# SOL helper (pure-math paths are import-safe anywhere; ship alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

# ---- timed regime: empirically chosen length (dense O(T^2) baseline; headroom grows with T) ----
B_T = 1
T_T = 2048
HQ_T = 8         # query heads
H_T = 8          # kv heads
K_T = 128        # head dim
V_T = 128
WINDOW_T = 1024  # sliding-window width (causal)
DECAY = 0.05     # per-channel log-decay magnitude (g in [-DECAY, 0])
DTYPE = torch.bfloat16
REL_MAX_TOL = 3e-2
REL_L2_TOL = 1.5e-2
WARMUP = 3
ITERS = 10
RCP_LN2 = 1.4426950408889634

# ---- hidden correctness suite: many diverse shapes (discrimination) ----
# (B, T, HQ, H, K, V, W). The baked baseline is a DENSE O(T^2) reference, so the suite
# stays within dense-tractable context lengths (<=1024) while spanning power-of-two and
# NON-power-of-two T -> causal + window boundary at varying context. W (window) varies
# incl. W>=T (full causal) and small windows -> sliding-window skip logic. HQ/H vary
# incl. GQA (HQ>H) -> head-group broadcast. B and K/V vary. A kernel correct only for
# the single timed shape (or only full-causal, or only HQ==H) fails these.
CORR_SHAPES = [
    (1, 256, 8, 8, 128, 128, 128),   (2, 256, 8, 8, 64, 64, 256),
    (1, 384, 8, 2, 128, 128, 192),   (1, 512, 4, 4, 128, 128, 512),
    (2, 320, 8, 4, 64, 64, 160),     (1, 640, 8, 8, 128, 128, 256),
    (1, 448, 4, 1, 128, 128, 448),   (2, 384, 8, 8, 64, 128, 128),
    (1, 768, 8, 2, 128, 128, 384),   (1, 512, 8, 8, 64, 64, 1024),
    (1, 1024, 8, 8, 128, 128, 512),  (2, 192, 4, 4, 128, 128, 96),
]


def build_inputs(B, T, HQ, H, K, V, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=gen)
    q = r(B, T, HQ, K)
    k = r(B, T, H, K)
    v = r(B, T, H, V)
    # Per-channel log-decay g of shape [B, T, HQ, K]; gentle NEGATIVE decay keeps
    # exp2(P_i - P_j) <= 1 over the causal/windowed region (fp32-safe).
    g = (-DECAY * torch.rand(B, T, HQ, K, device=device, generator=gen)).to(DTYPE)
    return dict(q=q, k=k, v=v, g=g)


def run_scope(inp, window):
    return parallel_wall_attn(inp["q"], inp["k"], inp["v"], inp["g"], window_size=window)


def wall_reference(inp, window):
    """Independent trusted fp32 windowed decay-softmax attention (ground truth; NOT in
    scope, and NOT the shipped naive path). Chunked over query blocks to bound memory."""
    q = inp["q"].float(); k = inp["k"].float(); v = inp["v"].float(); g = inp["g"].float()
    B, T, HQd, D = q.shape
    Hd = k.shape[2]; G = HQd // Hd
    Vd = v.shape[-1]
    scale = D ** -0.5
    W = window
    P = torch.cumsum(g, dim=1) * RCP_LN2                     # [B, T, HQ, K]
    k_exp = k.repeat_interleave(G, dim=2)                    # [B, T, HQ, K]
    v_exp = v.repeat_interleave(G, dim=2)                    # [B, T, HQ, V]
    o = q.new_zeros(B, T, HQd, Vd)
    BQ = 256
    for s in range(0, T, BQ):
        e = min(s + BQ, T)
        lo = max(0, s - W + 1) if W is not None else 0
        Pq = P[:, s:e]
        Pk = P[:, lo:e]
        qb = q[:, s:e]
        kb = k_exp[:, lo:e]
        vb = v_exp[:, lo:e]
        diff = Pq.unsqueeze(2) - Pk.unsqueeze(1)             # [B, bq, bk, HQ, K]
        sc = (qb.unsqueeze(2) * kb.unsqueeze(1) * torch.exp2(diff)).sum(-1)  # [B, bq, bk, HQ]
        sc = (sc * (scale * RCP_LN2)).permute(0, 3, 1, 2)    # [B, HQ, bq, bk]
        i_idx = torch.arange(s, e, device=q.device).view(1, 1, -1, 1)
        j_idx = torch.arange(lo, e, device=q.device).view(1, 1, 1, -1)
        valid = (j_idx <= i_idx)
        if W is not None:
            valid = valid & (i_idx - j_idx < W)
        sc = sc.masked_fill(~valid, float("-inf"))
        m = sc.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp2(sc - m)
        w = p / p.sum(-1, keepdim=True)
        ob = torch.einsum('bhij,bjhv->bihv', w, vb)          # [B, bq, HQ, V]
        o[:, s:e] = ob
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
    for i, (B, T, HQ, H, K, V, W) in enumerate(CORR_SHAPES):
        inp = build_inputs(B, T, HQ, H, K, V, seed=i)
        tag = f"{B}x{T}xHQ{HQ}xH{H}x{K}x{V}xW{W}"
        try:
            o = run_scope(inp, W).float()
        except Exception as e:
            detail[tag] = {"error": type(e).__name__ + ":" + str(e)[:70], "passed": False}
            continue
        ref = wall_reference(inp, W)
        torch.cuda.synchronize()
        if list(o.shape) != [B, T, HQ, V]:
            detail[tag] = {"shape": list(o.shape), "passed": False}
            continue
        rel_max, rel_l2, _ = _relnorm(o, ref)
        ok = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        n_pass += int(ok)
        detail[tag] = {"rel_max": round(rel_max, 5), "rel_l2": round(rel_l2, 5), "passed": bool(ok)}
    total = len(CORR_SHAPES)
    frac = n_pass / total
    print("WRO_WALLATTN_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    inp = build_inputs(B_T, T_T, HQ_T, H_T, K_T, V_T, seed=0)
    for _ in range(WARMUP):
        run_scope(inp, WINDOW_T)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run_scope(inp, WINDOW_T)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS,
           "batch": B_T, "seqlen": T_T, "hq": HQ_T, "h": H_T, "k": K_T, "v": V_T,
           "window": WINDOW_T}
    # SOL. The useful windowed attention does, per query and query head,
    # a QK^T and PV over ~min(T,W) keys:  FLOPs ~ 2 * B*HQ*T*W*(K+V) (windowed). The
    # compute-optimal fused kernel streams the window with an online softmax; the
    # roofline ceiling for that arithmetic is the SOL anchor. The degraded dense form
    # instead does O(T^2) work + materializes the full score tensor -> far below SOL.
    w_eff = min(T_T, WINDOW_T)
    flops = 2.0 * B_T * HQ_T * T_T * w_eff * (K_T + V_T)
    # bytes: read q,g [B,T,HQ,K] + k,v [B,T,H,*] + write o (windowed reuse bounded)
    io = B_T * T_T * HQ_T * (K_T + K_T) + B_T * T_T * H_T * (K_T + V_T) + B_T * T_T * HQ_T * V_T
    bytes_moved = io * 2  # bf16 lower bound
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
    print("WRO_WALLATTN_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_WALLATTN_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_WALLATTN_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
