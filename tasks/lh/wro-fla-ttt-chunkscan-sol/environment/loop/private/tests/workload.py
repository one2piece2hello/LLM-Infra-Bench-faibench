#!/usr/bin/env python3
"""Standalone workload for the Test-Time-Training (TTT) linear chunk forward subsystem
(``fla.ops.ttt.chunk_ttt_linear``).

Drives the PUBLIC entry ``chunk_ttt_linear(q, k, v, w, b, eta, ...)`` (imported from the
baked /app/repo tree). TTT-linear treats each mini-batch of tokens as an inner-loop test-
time gradient step on a linear "fast weight" state under a layer-norm reconstruction
objective, then reads out through the updated state and an output layer norm.

Two modes:

  correctness : run the subsystem over a DIVERSE hidden suite of shapes (varying T incl.
                non-multiple-of-16 lengths that exercise padding, varying B/H/D) and
                compare each against an INDEPENDENT fp32 TTT-linear reference computed
                here (NOT part of the editable scope). Emits a pass FRACTION over the
                whole suite (graded correctness).
  timing      : warmup + timed repeats of the subsystem call only, on the big long-
                sequence regime; also reports sol_fraction against the H20 roofline. The
                degraded form walks the sequence one mini-batch at a time -> the wall time
                grows with the number of dependent mini-batch steps (headroom grows with T).

Emits one line ``WRO_TTT_RESULT {json}``.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from fla.ops.ttt import chunk_ttt_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    _HAVE_H20 = True
except Exception:
    _HAVE_H20 = False

B_T = 8
T_T = 4096
H_T = 8
D_T = 128
CS = 16
DTYPE = torch.bfloat16
EPS = 1e-6
REL_MAX_TOL = 3e-2
REL_L2_TOL = 1.5e-2
REL_MAX_TOL_H = 6e-2
REL_L2_TOL_H = 3e-2
WARMUP = 3
ITERS = 10

# (B, T, H, D). T mixes multiples of 16 and NON-multiples (200,360,520,...) exercising the
# pad path; B in {1..4}, H in {1,2,4,8}; D in {64,128}. A kernel correct only for the timed
# shape (or only multiple-of-16 T) fails these.
CORR_SHAPES = [
    (2, 128, 2, 64),    (1, 192, 8, 128),
    (2, 256, 4, 64),    (3, 320, 1, 64),
    (1, 200, 8, 128),   (2, 512, 2, 128),
    (4, 128, 4, 128),   (1, 360, 2, 64),
    (2, 528, 8, 128),   (1, 1024, 4, 64),
    (2, 768, 1, 64),    (1, 640, 8, 128),
]


def build_inputs(B, T, H, D, seed=0, device="cuda"):
    g_ = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=g_)
    q = r(B, T, H, D) * 0.5
    k = r(B, T, H, D) * 0.5
    v = r(B, T, H, D) * 0.5
    w = (1.0 + 0.1 * r(H, D)).to(DTYPE)
    b = (0.1 * r(H, D)).to(DTYPE)
    eta = (torch.rand(B, T, H, 1, device=device, dtype=torch.float32, generator=g_) * 0.1).to(DTYPE)
    return dict(q=q, k=k, v=v, w=w, b=b, eta=eta)


def run_scope(inp, output_final_state=True):
    o, ht, htb = chunk_ttt_linear(
        q=inp["q"], k=inp["k"], v=inp["v"], w=inp["w"], b=inp["b"], eta=inp["eta"],
        scale=None, eps=EPS, chunk_size=CS,
        initial_state=None, initial_state_bias=None,
        output_final_state=output_final_state,
    )
    return o, ht, htb


def ttt_reference(inp):
    """Independent trusted fp32 TTT-linear scan (ground truth; NOT in the editable
    scope). Mirrors fla's chunk_ttt_linear_ref / ttt_linear semantics."""
    q = inp["q"].transpose(1, 2).float()
    k = inp["k"].transpose(1, 2).float()
    v = inp["v"].transpose(1, 2).float()
    eta = inp["eta"].transpose(1, 2).float()
    w = inp["w"].float(); b = inp["b"].float()
    B, H, T, D = q.shape
    BT = CS
    scale = D ** -0.5
    padded = (BT - (T % BT)) % BT
    if padded:
        q = F.pad(q, (0, 0, 0, padded)); k = F.pad(k, (0, 0, 0, padded))
        v = F.pad(v, (0, 0, 0, padded)); eta = F.pad(eta, (0, 0, 0, padded))
        eta[:, :, -1, :] = eta[:, :, -(padded + 1), :]
    Tp = q.shape[-2]
    NT = Tp // BT
    _q = q.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _k = k.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _v = v.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _eta = eta.reshape(B, H, NT, BT, 1).permute(2, 0, 1, 3, 4)
    wr = w.reshape(H, 1, D); br = b.reshape(H, 1, D)
    hstate = torch.zeros((B, H, D, D), device=v.device, dtype=torch.float32)
    hb = torch.zeros((B, H, 1, D), device=v.device, dtype=torch.float32)
    qq = _q * scale
    o = torch.empty_like(_v)
    for i in range(NT):
        q_i, k_i, v_i, eta_i = qq[i], _k[i], _v[i], _eta[i]
        kh = k_i @ hstate + hb
        rec = v_i - k_i
        mean = kh.mean(-1, True); var = kh.var(-1, unbiased=False, keepdim=True)
        rstd = torch.sqrt(var + EPS); kh_hat = (kh - mean) / rstd
        gg = (wr * kh_hat + br - rec) * wr
        v_new = (D * gg - gg.sum(-1, True) - kh_hat * (gg * kh_hat).sum(-1, True)) / (rstd * D)
        Attn = torch.tril(q_i @ k_i.transpose(-2, -1))
        o_i = q_i @ hstate - (eta_i * Attn) @ v_new + hb - torch.tril(eta_i.expand_as(Attn)) @ v_new
        hstate = hstate - (eta_i[:, :, -1, :, None] * k_i).transpose(-1, -2) @ v_new
        hb = hb - torch.sum(eta_i[:, :, -1, :, None] * v_new, dim=-2, keepdim=True)
        mean = o_i.mean(-1, True); var = o_i.var(-1, unbiased=False, keepdim=True)
        rstd = torch.sqrt(var + EPS)
        o[i] = o_i + (o_i - mean) / rstd * wr + br
    o = o.permute(1, 2, 0, 3, 4).reshape(B, H, Tp, D)[:, :, :T, :].transpose(1, 2)
    return o, hstate


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def correctness():
    torch.cuda.synchronize()
    n_pass = 0
    detail = {}
    for i, (B, T, H, D) in enumerate(CORR_SHAPES):
        inp = build_inputs(B, T, H, D, seed=i)
        tag = f"{B}x{T}x{H}x{D}"
        try:
            o, ht, htb = run_scope(inp, output_final_state=True)
            o = o.float()
        except Exception as e:
            detail[tag] = {"error": type(e).__name__ + ":" + str(e)[:70], "passed": False}
            continue
        ref_o, ref_h = ttt_reference(inp)
        torch.cuda.synchronize()
        if list(o.shape) != [B, T, H, D]:
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
    print("WRO_TTT_RESULT " + json.dumps({
        "mode": "correctness", "correctness_ok": (n_pass == total),
        "correctness_frac": round(frac, 4), "passed": n_pass, "total": total,
        "detail": detail}))
    sys.exit(0 if n_pass == total else 3)


def timing():
    torch.cuda.synchronize()
    inp = build_inputs(B_T, T_T, H_T, D_T, seed=0)
    for _ in range(WARMUP):
        run_scope(inp, output_final_state=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run_scope(inp, output_final_state=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    res = {"mode": "timing", "timing_ms": ms, "iters": ITERS,
           "batch": B_T, "seqlen": T_T, "heads": H_T, "d": D_T, "chunk_size": CS}
    # SOL. TTT per mini-batch / (batch,head): reconstruction + inner-loop
    # gradient + dual matmuls dominated by q@h (BT*D*D), k@h (BT*D*D), Attn@v_new
    # (BT*BT*D), state update (BT*D*D) -> ~4*BT*D*D + O(BT^2 D) MACs/step over NT=T/BT
    # steps -> ~4*B*H*T*D*D. Chunk-parallel batches the NT steps into fewer larger matmuls.
    flops = 2.0 * B_T * H_T * T_T * 4.0 * (D_T * D_T)
    io = B_T * T_T * H_T * (D_T + D_T + D_T) + B_T * T_T * H_T * D_T
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
    print("WRO_TTT_RESULT " + json.dumps(res))
    sys.exit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_TTT_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_TTT_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
