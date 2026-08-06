#!/usr/bin/env python3
"""Standalone workload for the Gated DeltaNet (gated delta-rule linear attention)
chunk forward subsystem (``fla.ops.gated_delta_rule.chunk_gated_delta_rule``).

Drives the PUBLIC entry ``chunk_gated_delta_rule(q, k, v, g, beta, ...)`` (imported
from the baked /app/repo tree) with synthetic fixed-seed tensors on the GPU. The op
maintains a K x V state with BOTH a per-step scalar decay gate g AND the delta-rule
error-correcting update:

    h_t = diag(exp(g_t)) h_{t-1}
    u_t = beta_t * (v_t - h_t^T k_t)
    h_t = h_t + k_t (x) u_t
    o_t = h_t^T (q_t * scale)

Two modes:

  correctness : run the subsystem over a DIVERSE hidden suite of shapes (varying T incl.
                non-power-of-two multi-chunk lengths, varying B/H, varying head width
                K/V) and compare each against an INDEPENDENT fp32 sequential gated
                delta-rule reference computed here (NOT part of the editable scope), by
                relative-norm tolerance. Emits a pass FRACTION over the whole suite
                (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on the big
                long-sequence regime; also reports sol_fraction against the H20
                roofline. The degraded form runs the gated-delta scan one time step at
                a time over T -> the wall time grows with T and sits far below the
                roofline (headroom grows with T).

Emits one line ``WRO_GDN_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

# ---- timed regime: a long sequence (per-step gated-delta recurrence; headroom grows with T) ----
B_T = 4
T_T = 4096
H_T = 8
K_T = 128
V_T = 128
DTYPE = torch.bfloat16
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
REL_MAX_TOL_H = 5e-2
REL_L2_TOL_H = 2e-2
WARMUP = 3
ITERS = 10

# ---- hidden correctness suite: many diverse shapes (discrimination) ----
# (B, T, H, K, V). T mixes power-of-two (128,256,512,1024) and NON-power-of-two multi-
# chunk (192,320,384,768,960,1536) lengths (chunk_size=64) -> inter-chunk state carry
# across a varying #chunks. B in {1..4}, H in {1,2,4,8}; K/V in {64,128}. A kernel
# correct only for the single timed shape (or only power-of-two T) fails these.
CORR_SHAPES = [
    (2, 128, 2, 64, 64),    (1, 192, 8, 128, 128),
    (2, 256, 4, 128, 64),   (3, 320, 1, 64, 64),
    (1, 384, 8, 128, 128),  (2, 512, 2, 64, 128),
    (4, 128, 4, 128, 128),  (1, 768, 2, 64, 64),
    (2, 960, 8, 128, 128),  (1, 1024, 4, 128, 64),
    (2, 1536, 1, 64, 64),   (1, 640, 8, 128, 128),
]


def build_inputs(B, T, H, K, V, seed=0, device="cuda"):
    g_ = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=g_)
    q = r(B, T, H, K)
    k = torch.nn.functional.normalize(r(B, T, H, K).float(), p=2, dim=-1).to(DTYPE)
    v = r(B, T, H, V)
    # g: log-space decay in (-inf, 0]; logsigmoid keeps the recurrence contractive.
    g = torch.nn.functional.logsigmoid(torch.rand(B, T, H, device=device,
                                                  dtype=torch.float32, generator=g_)).to(DTYPE)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32, generator=g_).sigmoid().to(DTYPE)
    return dict(q=q, k=k, v=v, g=g, beta=beta)


def run_scope(inp, output_final_state=True):
    o, ht = chunk_gated_delta_rule(
        q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"], beta=inp["beta"],
        scale=None, initial_state=None, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
    )
    return o, ht


def gdn_reference(inp):
    """Independent trusted fp32 sequential gated delta-rule scan (ground truth; NOT in
    the editable scope)."""
    q = inp["q"].float(); k = inp["k"].float(); v = inp["v"].float()
    g = inp["g"].float(); beta = inp["beta"].float()
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** -0.5
    q = q * scale
    o = q.new_zeros(b, t, h, dv)
    state = q.new_zeros(b, h, dk, dv)
    for i in range(t):
        state = state * torch.exp(g[:, i])[..., None, None]
        u = v[:, i] - (state * k[:, i][..., None]).sum(-2)
        u = u * beta[:, i][..., None]
        state = state + k[:, i][..., None] * u[..., None, :]
        o[:, i] = (state * q[:, i][..., None]).sum(-2)
    return o, state


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def correctness():
    torch.cuda.synchronize()
    n_pass = 0
    detail = {}
    for i, (B, T, H, K, V) in enumerate(CORR_SHAPES):
        inp = build_inputs(B, T, H, K, V, seed=i)
        tag = f"{B}x{T}x{H}x{K}x{V}"
        try:
            o, ht = run_scope(inp, output_final_state=True)
            o = o.float()
        except Exception as e:
            detail[tag] = {"error": type(e).__name__ + ":" + str(e)[:70], "passed": False}
            continue
        ref_o, ref_h = gdn_reference(inp)
        torch.cuda.synchronize()
        if list(o.shape) != [B, T, H, V]:
            detail[tag] = {"shape": list(o.shape), "passed": False}
            continue
        rel_max_o, rel_l2_o, _ = _relnorm(o, ref_o)
        ok = (rel_max_o <= REL_MAX_TOL) and (rel_l2_o <= REL_L2_TOL)
        rec = {"rel_max_o": round(rel_max_o, 5), "rel_l2_o": round(rel_l2_o, 5)}
        if ht is not None:
            rel_max_h, rel_l2_h, _ = _relnorm(ht.float(), ref_h)
            ok = ok and (rel_max_h <= REL_MAX_TOL_H) and (rel_l2_h <= REL_L2_TOL_H)
            rec["rel_max_h"] = round(rel_max_h, 5)
            rec["rel_l2_h"] = round(rel_l2_h, 5)
        rec["passed"] = bool(ok)
        n_pass += int(ok)
        detail[tag] = rec
    total = len(CORR_SHAPES)
    frac = n_pass / total
    print("WRO_GDN_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    inp = build_inputs(B_T, T_T, H_T, K_T, V_T, seed=0)
    for _ in range(WARMUP):
        run_scope(inp, output_final_state=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run_scope(inp, output_final_state=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS,
           "batch": B_T, "seqlen": T_T, "heads": H_T, "k": K_T, "v": V_T}
    # SOL. Gated delta-rule per step / (batch,head): decay scale (K*V),
    # h^T k (K*V), rank-1 update (K*V), readout h^T q (K*V) -> ~4*K*V mul-adds/step.
    # Total FLOPs = 2 * B*H*T * 4*K*V. Chunk-parallel batches into matmuls; eager form
    # is latency/step bound (T serial launches).
    flops = 2.0 * B_T * H_T * T_T * 4.0 * (K_T * V_T)
    io = B_T * T_T * H_T * (K_T + K_T + V_T) + B_T * T_T * H_T * V_T
    bytes_moved = io * 2
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
    print("WRO_GDN_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_GDN_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_GDN_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
