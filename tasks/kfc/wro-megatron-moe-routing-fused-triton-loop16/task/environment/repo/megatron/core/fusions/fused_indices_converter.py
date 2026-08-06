# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


def _indices_to_multihot_eager(indices, probs_indices, num_of_local_experts):
    """SLOW-BUT-CORRECT reference for the indices->multihot conversion.

    A straightforward eager PyTorch scatter: for every (token, slot) pair whose
    index is valid (not the -1 drop marker and inside the local expert range),
    write a 1 into the multihot map and the slot's probability into the multihot
    probs, recording which slot filled each expert (for the reverse pass). Correct
    for any input; it just issues several boolean-index / scatter launches and a
    device-dependent gather rather than a single fused pass.

    Args:
        indices: [num_of_tokens, topk] int token->expert indices (-1 == masked out).
        probs_indices: [num_of_tokens, topk] per-slot probabilities.
        num_of_local_experts: int.

    Returns:
        multihot_indices: [num_of_tokens, num_of_local_experts] (bool)
        probs_in_multihot: [num_of_tokens, num_of_local_experts] (probs_indices.dtype)
        position_map: [num_of_tokens, num_of_local_experts] (int32, -1 default) — the
            slot in `indices` that filled each expert; used by the reverse pass.
    """
    num_of_tokens, topk = indices.shape
    device = indices.device

    multihot_indices = torch.zeros(
        (num_of_tokens, num_of_local_experts), dtype=torch.long, device=device
    )
    probs_in_multihot = torch.zeros(
        (num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device=device
    )
    position_map = torch.full(
        (num_of_tokens, num_of_local_experts), -1, dtype=torch.int32, device=device
    )

    # Valid slots: not the drop marker AND inside the local expert range.
    mask = (indices != -1) & (indices < num_of_local_experts)
    valid_indices = indices[mask]
    counts = mask.sum(dim=1)
    row_indices = torch.arange(num_of_tokens, device=device).repeat_interleave(counts)
    # topk-slot position of each valid entry (for the reverse pass).
    slot_positions = torch.arange(topk, device=device).expand(num_of_tokens, topk)[mask]

    multihot_indices[row_indices, valid_indices] = 1
    probs_in_multihot[row_indices, valid_indices] = probs_indices[mask]
    position_map[row_indices, valid_indices] = slot_positions.to(torch.int32)

    return multihot_indices.bool(), probs_in_multihot, position_map


class IndicesToMultihot(torch.autograd.Function):
    """Convert moe topk indices to multihot representation.

    This class implements a custom forward and backward propagation
    operation for converting indices to multihot representation.
    It is an experimental feature and may change in future versions.
    """

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        '''Forward function for IndicesToMultihot

        Convert indices to multihot representation.

        Args:
            indices: [num_of_tokens, topk]
            probs_indices: [num_of_tokens, topk]
            num_of_local_experts: int

        Returns:
            multihot_indices: [num_of_tokens, num_of_local_experts]
            probs_in_multihot: [num_of_tokens, num_of_local_experts]
        '''
        assert (
            indices.shape == probs_indices.shape
        ), "indices and probs_indices must have the same shape"
        multihot_indices, probs_in_multihot, position_map = _indices_to_multihot_eager(
            indices, probs_indices, num_of_local_experts
        )
        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = indices.shape[0]
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = indices.shape[1]
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        '''Backward function for IndicesToMultihot

        Convert multihot probs representation back to indices.
        indices is ignored in the backward function.

        Args:
            grad_multihot_indices: [num_of_tokens, num_of_local_experts]
            grad_probs_in_multihot: [num_of_tokens, num_of_local_experts]

        Returns:
            grad_probs_indices: [num_of_tokens, topk]
        '''
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        grad_probs_indices = torch.zeros(
            (num_of_tokens, topk), dtype=grad_probs_in_multihot.dtype, device=position_map.device
        )
        # For each expert whose position_map entry is a valid slot, route the
        # multihot-probs gradient back to that slot in the indices layout.
        valid = position_map != -1
        rows = torch.arange(num_of_tokens, device=position_map.device).unsqueeze(1).expand(
            num_of_tokens, num_of_local_experts
        )[valid]
        slots = position_map[valid].long()
        grad_probs_indices[rows, slots] = grad_probs_in_multihot[valid]
        return None, grad_probs_indices, None, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert moe topk indices to multihot representation.

    This function is an experimental feature and may change in future versions.
    """
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)
