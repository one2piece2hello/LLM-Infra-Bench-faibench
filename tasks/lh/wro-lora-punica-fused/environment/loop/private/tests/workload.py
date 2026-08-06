#!/usr/bin/env python3
"""Standalone workload for the punica LoRA shrink/expand subsystem. Drives the
registered ops (lora_shrink -> lora_expand) with synthetic multi-LoRA batched
inputs, using EXACTLY vLLM's own convention (tests/lora/test_punica_ops.py):
3D lora weights, fp32 shrink buffer, LoRAKernelMeta routing, cleared ptr caches.
Correctness vs an INDEPENDENT fp32 per-LoRA reference (NOT in scope) by relative-norm;
timing measures the two ops. Emits WRO_LORA_RESULT."""
import json
import sys
import time

import torch

from vllm.lora.ops.triton_ops.lora_shrink_op import lora_shrink
from vllm.lora.ops.triton_ops.lora_expand_op import lora_expand
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT

M = 4096            # tokens
HIDDEN = 4096
OUT = 4096
NUM_LORAS = 8
RANK = 16
NSLICES = 1
SCALING = 0.5
DTYPE = torch.float16
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
WARMUP = 3
ITERS = 10


def build_inputs(seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.rand(M, HIDDEN, device=device, dtype=DTYPE, generator=g) - 0.5)          # (M, hidden)
    lora_a = [(torch.rand(NUM_LORAS, RANK, HIDDEN, device=device, dtype=DTYPE, generator=g) - 0.5)
              for _ in range(NSLICES)]                                                   # 3D (num_loras, rank, hidden)
    lora_b = [(torch.rand(NUM_LORAS, OUT, RANK, device=device, dtype=DTYPE, generator=g) - 0.5)
              for _ in range(NSLICES)]                                                   # 3D (num_loras, out, rank)
    tlm = torch.randint(0, NUM_LORAS, (M,), device=device, dtype=torch.int32, generator=g)
    meta = LoRAKernelMeta.make(max_loras=NUM_LORAS, max_num_tokens=M, device=device)
    meta.prepare_tensors(tlm)
    return dict(x=x, lora_a=lora_a, lora_b=lora_b, tlm=tlm, meta=meta)


def run_scope(inp):
    _LORA_A_PTR_DICT.clear(); _LORA_B_PTR_DICT.clear()
    buffer = torch.zeros(NSLICES, M, RANK, device=inp["x"].device, dtype=torch.float32)  # shrink out = fp32
    lora_shrink(inp["x"], inp["lora_a"], buffer, *inp["meta"].meta_args(M), SCALING)
    output = torch.zeros(M, OUT * NSLICES, device=inp["x"].device, dtype=inp["x"].dtype)
    lora_expand(buffer, inp["lora_b"], output, *inp["meta"].meta_args(M),
                offset_start=0, add_inputs=True)
    return output


def lora_reference(inp):
    """Independent fp32 per-LoRA shrink->expand (ground truth; NOT in scope)."""
    x = inp["x"].float()
    tlm = inp["tlm"]
    out = torch.zeros(M, OUT * NSLICES, device=x.device, dtype=torch.float32)
    for s in range(NSLICES):
        A = inp["lora_a"][s].float()   # (num_loras, rank, hidden)
        B = inp["lora_b"][s].float()   # (num_loras, out, rank)
        for lid in range(NUM_LORAS):
            mask = tlm == lid
            if not bool(mask.any()):
                continue
            buf = (x[mask] @ A[lid].t()) * SCALING     # (n_l, rank), scaling applied in shrink
            out[mask, s * OUT:(s + 1) * OUT] = buf @ B[lid].t()   # (n_l, out)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_LORA_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    torch.cuda.synchronize()
    inp = build_inputs(seed=0)
    if mode == "correctness":
        out = run_scope(inp).float()
        ref = lora_reference(inp)
        torch.cuda.synchronize()
        diff = (out - ref).abs()
        denom = ref.abs().max().clamp_min(1e-6)
        rel_max = float(diff.max() / denom)
        rel_l2 = float(diff.norm() / ref.norm().clamp_min(1e-6))
        ok = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        print("WRO_LORA_RESULT " + json.dumps({
            "mode": "correctness", "correctness_ok": bool(ok),
            "rel_max": rel_max, "rel_l2": rel_l2, "shape": list(out.shape)}))
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
        print("WRO_LORA_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS, "M": M, "num_loras": NUM_LORAS}))
        sys.exit(0)
    else:
        print("WRO_LORA_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
