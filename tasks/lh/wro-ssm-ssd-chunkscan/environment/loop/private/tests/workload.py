#!/usr/bin/env python3
"""Standalone workload for the Mamba-2 SSD state-space scan subsystem.

Drives the PUBLIC entry `mamba_chunk_scan_combined` (vllm mamba ssd ops) with
synthetic, fixed-seed tensors on the GPU. Two modes:

  correctness : run the subsystem, compare its output against an INDEPENDENT
                fp32 sequential ground-truth reference computed here (this
                reference is NOT part of the editable scope), assert allclose.
  timing      : warmup + timed repeats of the subsystem call only (paired
                measurement is done by the verifier across candidate/baseline).

Emits one line `WRO_SSM_RESULT {json}`.
"""
import json
import sys
import time

import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined,
)

# ---- workload configuration (the timed regime; large seqlen so the O(seqlen)
#      sequential recurrence separates clearly from the chunked reformulation,
#      yet finishes in a few seconds) ----
BATCH = 1
SEQLEN = 2048
NHEADS = 24
HEADDIM = 64
NGROUPS = 1
DSTATE = 128
CHUNK_SIZE = 256
DTYPE = torch.bfloat16
# bf16 GPU-kernel parity is judged by RELATIVE-NORM (not elementwise allclose,
# which is too strict on moderate-value elements given bf16 rounding between a
# chunked kernel and an fp32 sequential reference): relative max-abs and relative L2.
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
WARMUP = 3
ITERS = 10


def build_inputs(seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=DTYPE, generator=g)
    x = r(BATCH, SEQLEN, NHEADS, HEADDIM)
    dt = r(BATCH, SEQLEN, NHEADS)
    # A must be negative for a stable/contractive recurrence (A = -exp(a))
    A = -torch.exp(torch.randn(NHEADS, device=device, dtype=torch.float32,
                               generator=g))
    B = r(BATCH, SEQLEN, NGROUPS, DSTATE)
    C = r(BATCH, SEQLEN, NGROUPS, DSTATE)
    D = torch.randn(NHEADS, HEADDIM, device=device, dtype=torch.float32,
                    generator=g)
    z = r(BATCH, SEQLEN, NHEADS, HEADDIM)
    dt_bias = torch.randn(NHEADS, device=device, dtype=torch.float32,
                          generator=g)
    return dict(x=x, dt=dt, A=A, B=B, C=C, D=D, z=z, dt_bias=dt_bias)


def run_scope(inp):
    """Call the subsystem-under-test (candidate / degraded baseline code)."""
    out = torch.empty_like(inp["x"])
    mamba_chunk_scan_combined(
        inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], CHUNK_SIZE,
        D=inp["D"], z=inp["z"], dt_bias=inp["dt_bias"],
        dt_softplus=True, out=out, return_final_states=False)
    return out


def ssm_reference(inp):
    """Independent trusted fp32 sequential SSD scan (ground truth; NOT in the
    editable scope). Mirrors the Mamba-2 selective-scan semantics exactly:
        dt' = clamp(softplus(dt + dt_bias))
        h_t = exp(dt'_t * A) * h_{t-1} + dt'_t * (x_t outer B_t)
        y_t = (C_t . h_t) + D * x_t ; y *= silu(z)
    """
    import torch.nn.functional as F
    x = inp["x"].float(); dt = inp["dt"].float(); A = inp["A"].float()
    B = inp["B"].float(); C = inp["C"].float(); D = inp["D"].float()
    z = inp["z"].float(); dt_bias = inp["dt_bias"].float()
    b, l, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    dt = dt + dt_bias
    dt = torch.where(dt <= 20.0, F.softplus(dt), dt)
    dt = torch.clamp(dt, min=0.0)
    hpg = h // g
    idx = torch.arange(h, device=x.device) // hpg
    Bh = B.index_select(2, idx)  # (b,l,h,n)
    Ch = C.index_select(2, idx)
    state = torch.zeros(b, h, p, n, device=x.device, dtype=torch.float32)
    y = torch.empty(b, l, h, p, device=x.device, dtype=torch.float32)
    for t in range(l):
        dtt = dt[:, t, :]
        dA = torch.exp(dtt * A)
        dBx = dtt[:, :, None, None] * x[:, t, :, :, None] * Bh[:, t, :, None, :]
        state = dA[:, :, None, None] * state + dBx
        yt = (state * Ch[:, t, :, None, :]).sum(-1)
        yt = yt + D[None, :, :] * x[:, t, :, :]
        y[:, t, :, :] = yt
    y = y * (z * torch.sigmoid(z))
    return y


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_SSM_RESULT " + json.dumps({"error": "no_cuda"}))
        sys.exit(2)
    torch.cuda.synchronize()
    inp = build_inputs(seed=0)

    if mode == "correctness":
        out = run_scope(inp).float()
        ref = ssm_reference(inp)
        torch.cuda.synchronize()
        diff = (out - ref).abs()
        max_abs = float(diff.max())
        denom = ref.abs().max().clamp_min(1e-6)
        rel_max = float(max_abs / denom)
        rel_l2 = float(diff.norm() / ref.norm().clamp_min(1e-6))
        ok = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        print("WRO_SSM_RESULT " + json.dumps({
            "mode": "correctness", "correctness_ok": bool(ok),
            "rel_max": rel_max, "rel_l2": rel_l2, "max_abs_err": max_abs,
            "shape": list(out.shape)}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        for _ in range(WARMUP):
            run_scope(inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            run_scope(inp)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print("WRO_SSM_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS,
            "seqlen": SEQLEN, "chunk_size": CHUNK_SIZE}))
        sys.exit(0)
    else:
        print("WRO_SSM_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
