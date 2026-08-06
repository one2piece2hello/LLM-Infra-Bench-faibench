#!/usr/bin/env python3
"""Standalone workload for the Megatron-LM MoE routing-map padding subsystem
(megatron/core/transformer/moe/moe_utils.py :: pad_routing_map).

Drives the PUBLIC function `pad_routing_map` with synthetic, fixed-seed routing
maps on CPU (single process, no torch.distributed). Two modes:

  correctness : run the subsystem and compare the padded routing map against an
                INDEPENDENT vectorized reference computed here (NOT in scope).
  timing      : warmup + timed repeats of the `pad_routing_map` call only.

Emits one line `WRO_GDN_RESULT {json}`.

The timed regime uses many tokens (large N) and many experts (E) so the
per-(expert,token) scan of the slow baseline separates clearly from the
vectorized prefix-scan reformulation, and the gap GROWS with N.
"""
import json
import sys
import time

import torch

import megatron.core.transformer.moe.moe_utils as mu
from megatron.core.transformer.moe.moe_utils import pad_routing_map

# correctness: small & fast.
C_N, C_E, C_PAD = 128, 8, 16
# timing: many tokens + experts so the O(E*N) per-token scan dominates.
T_N, T_E, T_PAD = 6144, 64, 16
TOPK = 2
WARMUP = 1
ITERS = 3


def build_routing_map(N, E, topk, seed=0):
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(N, E, generator=g_)
    top_idx = scores.topk(topk, dim=1).indices
    rm = torch.zeros(N, E, dtype=torch.bool)
    rm.scatter_(1, top_idx, True)
    return rm


def pad_reference(routing_map, pad_multiple):
    """Independent vectorized reference: for each expert column, convert the
    earliest zeros to ones until the column's token count is a multiple of
    pad_multiple. NOT the editable scope."""
    rm = routing_map.clone().transpose(0, 1)          # [E, N]
    num_ones = rm.sum(dim=1)
    num_to_pad = (-num_ones) % pad_multiple
    is_zero = (rm == 0)
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)
    mask = zero_ranks <= num_to_pad.unsqueeze(1)
    rm[mask] = 1
    return rm.transpose(0, 1).contiguous()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"

    if mode == "correctness":
        rm = build_routing_map(C_N, C_E, TOPK, seed=0)
        out = pad_routing_map(rm.clone(), C_PAD)
        ref = pad_reference(rm, C_PAD)
        ok = bool(torch.equal(out.bool(), ref.bool()))
        # each expert column count must be a multiple of pad_multiple
        col_counts = out.bool().sum(dim=0)
        mult_ok = bool(int((col_counts % C_PAD == 0).all()))
        ok = ok and mult_ok
        res = {"mode": "correctness", "correctness_ok": bool(ok),
               "match_ref": bool(torch.equal(out.bool(), ref.bool())),
               "multiple_ok": mult_ok, "module": mu.__file__,
               "col_counts": [int(x) for x in col_counts.tolist()]}
        print("WRO_GDN_RESULT " + json.dumps(res))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        rm = build_routing_map(T_N, T_E, TOPK, seed=1)
        for _ in range(WARMUP):
            pad_routing_map(rm.clone(), T_PAD)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            pad_routing_map(rm.clone(), T_PAD)
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print("WRO_GDN_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS,
            "tokens": T_N, "experts": T_E, "pad_multiple": T_PAD}))
        sys.exit(0)
    else:
        print("WRO_GDN_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
