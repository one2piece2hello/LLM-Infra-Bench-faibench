#!/usr/bin/env python3
"""Standalone workload for the MoE routing-map / dispatch-indices preprocessing
subsystem (megatron.core.fusions).

Drives the two PUBLIC entry points that the DeepEP token dispatcher runs on its
preprocess chain, back to back as a coupled chain (this mirrors
`_DeepepManager.get_permuted_hidden_states_by_experts`: convert dispatch indices
to a multihot routing map, then pad that routing map to the FP8/FP4 alignment
multiple):

    from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
    from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

    routing_map, probs = fused_indices_to_multihot(indices, probs, num_local_experts)
    padded_map = fused_pad_routing_map(routing_map, pad_multiple)

Two modes:

  correctness : run the chain, compare its outputs (multihot routing map, its
                probabilities, and the padded map) against an INDEPENDENT eager
                reference computed here (NOT part of the editable scope). The
                integer maps must match EXACTLY; the gathered probabilities to a
                tiny float tolerance (pure gather, no arithmetic).
  timing      : warmup + timed repeats of the coupled chain only (paired
                measurement done by the verifier across candidate/baseline).

Emits one line `WRO_MOEROUTE_RESULT {json}`.

The timed regime uses large token/expert counts so the several eager-op launches
+ data-dependent (device-sync) gather of the slow baseline separate clearly from
the single-pass fused kernels, yet finish in well under a second.
"""
import json
import sys
import time

import torch

from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

# ---- workload configuration (the timed regime) ----
NUM_TOKENS = 16384        # tokens received on this rank after dispatch
TOPK = 8                  # experts each token routes to
NUM_LOCAL_EXPERTS = 256   # local experts on this rank
PAD_MULTIPLE = 32         # FP8/FP4 alignment multiple
DROP_FRAC = 0.05          # fraction of index slots masked out (-1), as in capacity drop
DTYPE = torch.float32     # DeepEP path uses float32 probs
WARMUP = 5
ITERS = 30


def build_inputs(seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    # each token selects TOPK distinct local experts (topk over random scores)
    scores = torch.rand(NUM_TOKENS, NUM_LOCAL_EXPERTS, device=device, generator=g)
    indices = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int64).contiguous()
    probs = torch.rand(NUM_TOKENS, TOPK, device=device, dtype=DTYPE, generator=g).contiguous()
    # emulate capacity-dropped tokens: mask a fraction of slots to -1 / 0.0
    dropmask = torch.rand(NUM_TOKENS, TOPK, device=device, generator=g) < DROP_FRAC
    indices = indices.masked_fill(dropmask, -1)
    probs = probs.masked_fill(dropmask, 0.0)
    return dict(indices=indices, probs=probs)


def run_scope(inp):
    """Call the subsystem-under-test (candidate / degraded baseline code): the
    coupled indices->multihot->pad preprocess chain."""
    routing_map, probs_multihot = fused_indices_to_multihot(
        inp["indices"], inp["probs"], NUM_LOCAL_EXPERTS
    )
    padded_map = fused_pad_routing_map(routing_map, PAD_MULTIPLE)
    return routing_map, probs_multihot, padded_map


def moe_route_reference(inp):
    """Independent trusted eager reference (ground truth; NOT in the editable
    scope). Mirrors the DeepEP preprocess chain exactly:
      - indices->multihot: scatter a 1 and the slot prob at each valid expert;
      - pad: flip the earliest zero slots of each expert until its token count is
        a multiple of PAD_MULTIPLE.
    """
    indices = inp["indices"]
    probs = inp["probs"]
    nt, topk = indices.shape
    device = indices.device
    ne = NUM_LOCAL_EXPERTS

    # indices -> multihot
    multihot = torch.zeros((nt, ne), dtype=torch.long, device=device)
    multihot_probs = torch.zeros((nt, ne), dtype=probs.dtype, device=device)
    mask = (indices != -1) & (indices < ne)
    valid = indices[mask]
    rows = torch.arange(nt, device=device).repeat_interleave(mask.sum(dim=1))
    multihot[rows, valid] = 1
    multihot_probs[rows, valid] = probs[mask]
    routing_map = multihot.bool()

    # pad the multihot routing map to a multiple of PAD_MULTIPLE per expert
    rm = routing_map.transpose(0, 1).contiguous().int()  # [ne, nt]
    num_ones = rm.sum(dim=1)
    num_to_pad = (-num_ones) % PAD_MULTIPLE
    is_zero = rm == 0
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)
    flip = (zero_ranks <= num_to_pad.unsqueeze(1)) & is_zero
    padded = torch.where(flip, torch.ones_like(rm), rm).transpose(0, 1)
    return routing_map, multihot_probs, padded


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_MOEROUTE_RESULT " + json.dumps({"error": "no_cuda"}))
        sys.exit(2)
    torch.cuda.synchronize()
    inp = build_inputs(seed=0)

    if mode == "correctness":
        rm, pm, padded = run_scope(inp)
        ref_rm, ref_pm, ref_padded = moe_route_reference(inp)
        torch.cuda.synchronize()
        rm_ok = bool(torch.equal(rm.long(), ref_rm.long()))
        padded_ok = bool(torch.equal(padded.long(), ref_padded.long()))
        probs_max_abs = float((pm.float() - ref_pm.float()).abs().max())
        probs_ok = probs_max_abs <= 1e-6
        ok = rm_ok and padded_ok and probs_ok
        res = {"mode": "correctness", "correctness_ok": bool(ok),
               "routing_map_ok": rm_ok, "padded_map_ok": padded_ok,
               "probs_ok": probs_ok, "probs_max_abs": probs_max_abs,
               "rm_shape": list(rm.shape), "padded_shape": list(padded.shape)}
        print("WRO_MOEROUTE_RESULT " + json.dumps(res))
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
        print("WRO_MOEROUTE_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS,
            "num_tokens": NUM_TOKENS, "topk": TOPK,
            "num_local_experts": NUM_LOCAL_EXPERTS, "pad_multiple": PAD_MULTIPLE}))
        sys.exit(0)
    else:
        print("WRO_MOEROUTE_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
