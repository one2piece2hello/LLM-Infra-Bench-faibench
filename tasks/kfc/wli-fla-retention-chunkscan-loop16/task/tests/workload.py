#!/usr/bin/env python3
"""Workload for multi-scale retention chunked linear attention
(fla.ops.retention.chunk_retention).

Scored regime: regular batched inputs [B, T, H, K] (no cu_seqlens, no
initial_state, output_final_state=False), so the subsystem reduces to the
causal decay-weighted retention output
    gamma[h] = 1 - 2 ** (-5 - h)
    o_t      = scale * sum_{s<=t} gamma[h]^(t-s) (q_t . k_s) v_s ,  scale = K**-0.5

  correctness : run chunk_retention; if it raises (unimplemented) or its output
                does not match an INDEPENDENT fp32 retention reference (computed
                here; NOT in the editable scope) within tolerance,
                correctness_ok=False.
  timing      : warmup + timed repeats of the subsystem call only.

Emits one line `WLI_RET_RESULT {json}`.
"""
import json
import sys
import time

import torch

from fla.ops.retention import chunk_retention

BATCH = 2
SEQLEN = 2048
H = 8               # heads
K = 128             # head dim (key/query)
V = 128             # head dim (value)
DTYPE = torch.bfloat16
REL_MAX_TOL = 3e-2
REL_L2_TOL = 1.5e-2
WARMUP = 3
ITERS = 10


def build_inputs(seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=gen)
    q = r(BATCH, SEQLEN, H, K)
    k = r(BATCH, SEQLEN, H, V)
    v = r(BATCH, SEQLEN, H, V)
    return dict(q=q, k=k, v=v)


def run_scope(inp):
    o, _ = chunk_retention(
        q=inp["q"], k=inp["k"], v=inp["v"],
        scale=None, initial_state=None, output_final_state=False,
    )
    return o


def retention_reference(inp):
    """Independent trusted fp32 retention reference (ground truth; NOT in scope).

    Chunked over query positions to bound memory. For each query position q and
    each key position s<=q:  weight = scale * gamma[h]^(q-s) * (q_q . k_s), then
    o_q = sum_s weight * v_s.  gamma[h] = 1 - 2**(-5-h),  scale = K**-0.5.
    """
    q = inp["q"].float().permute(0, 2, 1, 3)     # [B,H,T,K]
    k = inp["k"].float().permute(0, 2, 1, 3)     # [B,H,T,K]
    v = inp["v"].float().permute(0, 2, 1, 3)     # [B,H,T,V]
    B, Hh, T, Kk = q.shape
    Vv = v.shape[-1]
    scale = Kk ** -0.5
    dev = q.device
    hh = torch.arange(Hh, device=dev, dtype=torch.float)
    log_gamma = torch.log1p(-torch.pow(torch.tensor(2.0, device=dev), -5.0 - hh))  # log(gamma[h]) < 0
    pos = torch.arange(T, device=dev, dtype=torch.float)
    o = q.new_zeros(B, Hh, T, Vv)
    BQ = 512
    for s0 in range(0, T, BQ):
        s1 = min(s0 + BQ, T)
        qpos = pos[s0:s1].view(1, 1, -1, 1)                       # [1,1,bq,1]
        kpos = pos[:s1].view(1, 1, 1, -1)                         # [1,1,1,s1]
        dexp = (qpos - kpos)                                      # [1,1,bq,s1]
        causal = dexp >= 0
        # gamma[h]^(q-s) = exp((q-s)*log_gamma[h]); zero where s>q
        decay = torch.exp(dexp * log_gamma.view(1, Hh, 1, 1))     # [1,H,bq,s1]
        decay = decay * causal
        sc = torch.einsum('bhqk,bhsk->bhqs', q[:, :, s0:s1] * scale, k[:, :, :s1])  # [B,H,bq,s1]
        sc = sc * decay
        o[:, :, s0:s1] = torch.einsum('bhqs,bhsv->bhqv', sc, v[:, :, :s1])
    return o.permute(0, 2, 1, 3)                                  # [B,T,H,V]


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6)), max_abs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WLI_RET_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    torch.cuda.synchronize()
    inp = build_inputs(seed=0)

    if mode == "correctness":
        try:
            o = run_scope(inp).float()
        except NotImplementedError:
            print("WLI_RET_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False, "reason": "not_implemented"})); sys.exit(3)
        except Exception as e:
            print("WLI_RET_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False, "reason": "exception:" + type(e).__name__ + ":" + str(e)[:200]})); sys.exit(3)
        ref = retention_reference(inp)
        torch.cuda.synchronize()
        rel_max, rel_l2, max_abs = _relnorm(o, ref)
        ok = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        print("WLI_RET_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": bool(ok),
              "rel_max": rel_max, "rel_l2": rel_l2, "max_abs": max_abs, "shape": list(o.shape)}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        try:
            for _ in range(WARMUP):
                run_scope(inp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                run_scope(inp)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        except NotImplementedError:
            print("WLI_RET_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1, "reason": "not_implemented"})); sys.exit(3)
        print("WLI_RET_RESULT " + json.dumps({"mode": "timing", "timing_ms": ms, "iters": ITERS,
              "batch": BATCH, "seqlen": SEQLEN, "h": H, "k": K, "v": V}))
        sys.exit(0)
    else:
        print("WLI_RET_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
