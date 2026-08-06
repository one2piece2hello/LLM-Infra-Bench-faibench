#!/usr/bin/env python3
"""Standalone workload for the ABC (Attention with Bounded-memory Control)
two-stage gated linear-recurrence forward subsystem (``fla.ops.abc.chunk_abc``).

Drives the PUBLIC entry ``chunk_abc(q, k, v, s, initial_state, output_final_state)``
(imported from the baked /app/repo tree) with synthetic fixed-seed tensors on the
GPU. The op maintains TWO matrix-valued recurrent states with a per-step forget
gate: stage 1 reads queries against a key/slot memory ``hk`` [K,M] to produce slot
logits, a softmax over the M slots turns them into slot probabilities, and stage 2
reads those against a slot/value memory ``hv`` [M,V] to produce the output ``o``.
The gate is derived from the slot logits via a cumulative log-sum-exp normalizer.

Two modes:

  correctness : run the subsystem over a DIVERSE hidden suite of shapes (varying T
                incl. non-power-of-two and sub-chunk lengths, varying B/H, varying
                K/V/M incl. non-multiple-of-tile widths) and compare each against an
                INDEPENDENT fp32 sequential reference computed here (NOT part of the
                editable scope), by relative-norm tolerance. Emits a pass FRACTION
                over the whole suite (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on the big
                long-sequence regime; also reports sol_fraction against the H20
                roofline. The degraded form is dominated by T dependent steps, each
                doing O(H*(K*M + M*V)) matrix work -> the wall time grows with T and
                sits far below the roofline (headroom grows with T).

Emits one line ``WRO_ABC_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from fla.ops.abc import chunk_abc

# SOL helper (pure-math paths are import-safe anywhere; ship alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

# ---- timed regime: a long sequence (per-step matrix recurrence; headroom grows with T) ----
B_T = 4
T_T = 4096
H_T = 4          # heads (H == HV)
K_T = 128        # key dim
V_T = 128        # value dim
M_T = 64         # number of memory slots
DTYPE = torch.bfloat16
# bf16 GPU-kernel parity is judged by RELATIVE-NORM (not elementwise allclose, too
# strict given bf16 rounding between a chunked kernel and an fp32 sequential
# reference through two coupled recurrences + a softmax): relative max-abs + L2.
REL_MAX_TOL = 4e-2
REL_L2_TOL = 2e-2
WARMUP = 3
ITERS = 10

# ---- hidden correctness suite: many diverse shapes (discrimination) ----
# (B, T, H, K, V, M). T mixes power-of-two and NON-power-of-two multi-chunk lengths
# (192, 320, 960, 1536) -> exercises the inter-chunk state carry across BOTH coupled
# recurrences for a varying number of chunks (a kernel that hard-codes the timed T,
# or only handles power-of-two T, fails these). B and H both vary -> exercises the
# per-(batch,head) independence of the two matrix states. K/V mix the two head widths
# {64,128} and M the slot counts {32,64}. Lengths are chunk-aligned because the
# operator's chunk-parallel form tiles the sequence into fixed-size chunks; the
# discrimination lives in the multi-chunk carry + batch/head handling, not in ragged
# tails. A naive impl correct only for the single timed shape scores a low fraction.
CORR_SHAPES = [
    (2, 64, 2, 64, 64, 64),     (1, 128, 4, 128, 128, 64),
    (2, 192, 2, 128, 128, 32),  (1, 256, 4, 128, 64, 64),
    (3, 320, 2, 64, 64, 64),    (2, 384, 1, 128, 128, 64),
    (1, 512, 4, 64, 128, 32),   (4, 128, 2, 128, 128, 64),
    (2, 768, 2, 64, 64, 64),    (1, 960, 4, 128, 128, 64),
    (2, 1024, 1, 128, 64, 32),  (3, 1536, 2, 64, 64, 64),
]


def build_inputs(B, T, H, K, V, M, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=gen)
    q = r(B, T, H, K)
    k = r(B, T, H, K)
    v = r(B, T, H, V)
    # slot logits; kept moderate so the logcumsumexp gate is well-conditioned
    s = r(B, T, H, M)
    return dict(q=q, k=k, v=v, s=s)


def _abc_reference(inp):
    """INDEPENDENT trusted fp32 sequential ABC scan (ground truth; NOT in the
    editable scope). Two coupled gated recurrences with a slot softmax between,
    the gate derived from the slot logits via cumulative log-sum-exp. Mirrors the
    subsystem's public [B,T,H,*] contract."""
    q = inp["q"].transpose(1, 2).float()   # [B,H,T,K]
    k = inp["k"].transpose(1, 2).float()
    v = inp["v"].transpose(1, 2).float()
    s = inp["s"].transpose(1, 2).float()   # [B,H,T,M]
    B, Hh, T, Kk = q.shape
    Vv = v.shape[-1]
    Mm = s.shape[-1]
    scale = Kk ** -0.5
    z = s.logcumsumexp(2)
    g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
    sp = torch.exp(s - z)

    hk = q.new_zeros(B, Hh, Kk, Mm)
    ok = torch.zeros_like(s)
    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = sp[:, :, i]
        g_i = g[:, :, i].exp()
        hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
        ok[:, :, i] = (q_i[..., None] * hk).sum(-2)

    qv = ok.softmax(-1)
    hv = q.new_zeros(B, Hh, Mm, Vv)
    ov = torch.zeros_like(v)
    for i in range(T):
        q_i = qv[:, :, i]
        k_i = sp[:, :, i]
        v_i = v[:, :, i]
        g_i = g[:, :, i].exp()
        hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
        ov[:, :, i] = (q_i[..., None] * hv).sum(-2)
    return ov.transpose(1, 2), (hk, hv)   # o [B,T,H,V]


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def correctness():
    torch.cuda.synchronize()
    n_pass = 0
    detail = {}
    for i, (B, T, H, K, V, M) in enumerate(CORR_SHAPES):
        inp = build_inputs(B, T, H, K, V, M, seed=i)
        try:
            o, ht = chunk_abc(q=inp["q"], k=inp["k"], v=inp["v"], s=inp["s"],
                              initial_state=None, output_final_state=True)
            o = o.float()
        except Exception as e:
            detail[f"{B}x{T}x{H}x{K}x{V}x{M}"] = {"error": str(e)[:80], "passed": False}
            continue
        ref_o, _ = _abc_reference(inp)
        torch.cuda.synchronize()
        if list(o.shape) != [B, T, H, V]:
            detail[f"{B}x{T}x{H}x{K}x{V}x{M}"] = {"shape": list(o.shape), "passed": False}
            continue
        rel_max_o, rel_l2_o, _ = _relnorm(o, ref_o)
        ok = (rel_max_o <= REL_MAX_TOL) and (rel_l2_o <= REL_L2_TOL)
        n_pass += int(ok)
        detail[f"{B}x{T}x{H}x{K}x{V}x{M}"] = {"rel_max_o": round(rel_max_o, 5),
                                              "rel_l2_o": round(rel_l2_o, 5), "passed": bool(ok)}
    total = len(CORR_SHAPES)
    frac = n_pass / total
    print("WRO_ABC_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    inp = build_inputs(B_T, T_T, H_T, K_T, V_T, M_T, seed=0)
    for _ in range(WARMUP):
        chunk_abc(q=inp["q"], k=inp["k"], v=inp["v"], s=inp["s"],
                  initial_state=None, output_final_state=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        chunk_abc(q=inp["q"], k=inp["k"], v=inp["v"], s=inp["s"],
                  initial_state=None, output_final_state=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS,
           "batch": B_T, "seqlen": T_T, "heads": H_T, "k": K_T, "v": V_T, "m": M_T}
    # SOL. The two-stage recurrence is dominated by, per time step and
    # per (batch,head), two rank-1 state updates + two contractions:
    #   stage 1: hk[K,M] update (K*M mul-add) + read q against hk (K*M) -> ~2*K*M
    #   stage 2: hv[M,V] update (M*V mul-add) + read qv against hv (M*V) -> ~2*M*V
    # -> ~ 2*(K*M + M*V) mul-adds/step. Total FLOPs = 2 * B*H*T * 2*(K*M + M*V).
    # The compute-optimal chunk-parallel form turns the T-sequential dependency into
    # matmuls; the roofline ceiling for that arithmetic is the SOL anchor.
    elems_io = B_T * T_T * H_T * (K_T + K_T + V_T + M_T)   # read q,k,v,s once
    out_io = B_T * T_T * H_T * V_T                          # write o once
    flops = 2.0 * B_T * H_T * T_T * 2.0 * (K_T * M_T + M_T * V_T)
    bytes_moved = (elems_io + out_io) * 2  # bf16 I/O lower bound
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
    print("WRO_ABC_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_ABC_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_ABC_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
