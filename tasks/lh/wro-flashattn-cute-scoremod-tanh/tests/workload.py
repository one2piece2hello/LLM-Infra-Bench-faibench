#!/usr/bin/env python3
"""Standalone workload for the USER-PROVIDED gated-tanh score-modification multi-head attention
FORWARD subsystem (flash_attn.cute).

Drives the in-scope public entry ``flash_attn.cute.flash_attn_func`` imported from the baked
/app/repo flash-attention tree, exercising a workload-defined ``wro_scoremod_callable`` — a user
score_mod @cute.jit callable that applies the scalar-elementwise gated-tanh transform
``scores * (beta + (1-beta) * tanh(alpha*scores))`` (alpha=0.3, beta=0.5) to the scaled scores
BEFORE the optional causal mask and the softmax. Grouped-query attention (num_heads_kv <
num_heads_q) is supported by sharing each key/value head across consecutive query heads; cross-
attention shapes (seqlen_q != seqlen_k) are supported. This is the memory-bound operation whose
fused single-pass CuTe kernel this task is about.

Correctness is graded as a PASS-RATE over MANY diverse hidden cases (varied batch, sequence lengths
incl. cross-attention, head counts incl. grouped-query, head dims, causal on/off, dtype, and softmax
scale) against an INDEPENDENT fp32 reference (explicit materialized-score attention that applies
THE SAME gated-tanh transform to the scaled scores). Timing measures one forward over a long-
sequence workload (the memory-bound regime where the fused kernel avoids materializing the
O(seqlen^2) score matrix). Emits WRO_FA_RESULT.

Usage: python3 workload.py {correctness|timing}
"""
import json
import math
import sys
import time

import torch

from flash_attn.cute import flash_attn_func

# The user-provided score_mod callable is defined HERE at workload scope (uploaded fresh with
# tests per scoring run; never baked into the shipped image). The workload passes it to the entry point
# via `score_mod=wro_scoremod_callable` — this is the FlexAttention-style Callable API the task is
# about. On the fused path the CuTe kernel composes it inline in the online-softmax epilogue; the
# baked eager baseline applies the SAME gated-tanh transform in dense torch code.
import cutlass
import cutlass.cute as cute
from cutlass import Float32 as _WroFloat32

@cute.jit
def wro_scoremod_callable(scores, batch_idx, head_idx, q_idx=None, kv_idx=None,
                          seqlen_info=None, aux_tensors=None):
    t = cute.math.tanh(_WroFloat32(0.3) * scores, fastmath=True)
    return scores * (_WroFloat32(0.5) + _WroFloat32(0.5) * t)

# Module-defined scalars the score_mod uses. Must match wro_scoremod_callable exactly.
_WRO_ALPHA = 0.3
_WRO_BETA = 0.5

REL_L2_TOL = 3e-2
REL_MAX_TOL = 8e-2
WARMUP = 6
ITERS = 20
# Long-sequence memory-bound timing workload.
TIMING = dict(b=1, sq=8192, sk=8192, hq=8, hk=8, d=128, causal=True, dtype=torch.bfloat16)

# A diverse hidden correctness suite:
#   (b, sq, sk, hq, hk, d, causal, dtype, scale_mode)
# Every case genuinely exercises the score_mod: with alpha=0.3, beta=0.5 the gated-tanh transform
# `s * (beta + (1-beta) * tanh(alpha*s))` reshapes each score smoothly (identity at s=0, saturating
# outside |s|>~3/alpha ~= 10) and moves it away from the identity transform enough that on these
# workloads (verified on H20 pre-authoring), a fused call that DROPS the score_mod (score_mod=None)
# diverges by >= 0.26 rel-L2 on every shape/dtype — well beyond the 0.03 tolerance. scale_mode:
# "default" -> 1/sqrt(d); "custom" -> a non-default (smaller) scale.
CORR_CASES = [
    (2, 256, 256, 8, 8, 64, False, torch.bfloat16, "default"),
    (2, 256, 256, 8, 8, 64, True, torch.bfloat16, "default"),
    (1, 512, 512, 16, 16, 128, True, torch.bfloat16, "default"),
    (1, 512, 512, 16, 16, 128, False, torch.float16, "default"),
    (3, 384, 384, 12, 12, 64, True, torch.float16, "default"),
    (2, 640, 640, 8, 8, 128, True, torch.bfloat16, "custom"),
    (1, 1024, 1024, 16, 16, 128, True, torch.bfloat16, "default"),
    (2, 128, 512, 8, 8, 64, False, torch.bfloat16, "default"),   # cross-attn sq<sk
    (2, 512, 128, 8, 8, 64, False, torch.bfloat16, "default"),   # cross-attn sq>sk
    (1, 1024, 1024, 16, 4, 128, True, torch.bfloat16, "default"),  # GQA 4x
    (2, 512, 512, 16, 8, 128, True, torch.bfloat16, "default"),  # GQA 2x
    (1, 2048, 2048, 8, 8, 128, True, torch.bfloat16, "default"),  # long
    (1, 2048, 2048, 8, 8, 64, True, torch.bfloat16, "custom"),  # long + custom scale
    (2, 256, 256, 32, 32, 64, True, torch.bfloat16, "default"),  # many heads
    (1, 1536, 1536, 8, 2, 128, True, torch.bfloat16, "default"),  # GQA 4x
    (4, 192, 192, 8, 8, 128, False, torch.float16, "default"),
    (1, 4096, 4096, 8, 8, 128, True, torch.bfloat16, "default"),  # long causal
    (2, 384, 640, 8, 8, 64, False, torch.bfloat16, "custom"),   # cross-attn + custom
]


def _scale(d, mode):
    return (1.0 / math.sqrt(d)) if mode == "default" else (0.7 / math.sqrt(d))


def build(b, sq, sk, hq, hk, d, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    return q, k, v


def reference(q, k, v, causal, scale):
    """Independent fp32 reference: explicit materialized-score attention that applies the SAME
    gated-tanh transform ``scores * (beta + (1-beta) * tanh(alpha*scores))`` (alpha=0.3, beta=0.5)
    to the scale-multiplied scores, then the optional causal mask, then softmax and value-mix. GQA
    supported by repeating K/V heads."""
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
    scores = scores * (_WRO_BETA + (1.0 - _WRO_BETA) * torch.tanh(_WRO_ALPHA * scores))
    if causal:
        cm = torch.triu(torch.ones(sq, sk, device=scores.device, dtype=torch.bool),
                        diagonal=sk - sq + 1)
        scores = scores.masked_fill(cm, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vf).transpose(1, 2)
    return o  # (b, sq, hq, d) fp32


def rel_norms(cand, ref):
    cand = cand.to(torch.float32)
    ref = ref.to(torch.float32)
    denom = ref.norm().item() + 1e-9
    return ((cand - ref).norm().item() / denom,
            (cand - ref).abs().max().item() / (ref.abs().max().item() + 1e-9))


def run_scope(q, k, v, causal, scale):
    # Pass the module-defined score_mod callable through to the entry.
    o = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale,
                       score_mod=wro_scoremod_callable)
    return o[0] if isinstance(o, tuple) else o


def correctness():
    torch.cuda.init()
    n_pass = 0
    detail = {}
    for i, (b, sq, sk, hq, hk, d, causal, dtype, sm) in enumerate(CORR_CASES):
        try:
            scale = _scale(d, sm)
            q, k, v = build(b, sq, sk, hq, hk, d, dtype, seed=100 + i)
            out = run_scope(q, k, v, causal, scale)
            ref = reference(q, k, v, causal, scale)
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
    fn = lambda: run_scope(q, k, v, t["causal"], scale)
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
