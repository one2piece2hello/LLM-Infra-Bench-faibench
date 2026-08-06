# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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


def fused_pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Pad the MoE routing map so every expert's token count is a multiple of pad_multiple.

    SLOW-BUT-CORRECT reference implementation: a straightforward eager PyTorch
    formulation. For each expert it counts how many tokens are routed to it,
    computes how many extra tokens are needed to reach the next multiple of
    ``pad_multiple``, and flips that many of the expert's currently-unrouted
    (zero) token slots to 1, choosing the earliest slots. Correct for any input;
    it simply issues several elementwise/scan launches rather than a single pass.

    Args:
        routing_map (torch.Tensor): A boolean or integer tensor of shape [num_tokens,
            num_experts] indicating which tokens are routed to which experts.
        pad_multiple (int): The multiple to pad each expert's token count to.

    Returns:
        torch.Tensor: The padded routing map of shape [num_tokens, num_experts].
    """
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    # Work per-expert (row-wise) on the transposed map. contiguous().int() gives a
    # fresh buffer so the input tensor is never mutated (matches the entry contract).
    input_map = routing_map.transpose(0, 1).contiguous().int()  # [num_experts, num_tokens]

    # How many extra tokens each expert needs to hit the next multiple of pad_multiple.
    num_ones = input_map.sum(dim=1)
    num_to_pad = (-num_ones) % pad_multiple

    # Rank the zero slots in each expert row; flip the first `num_to_pad` of them.
    is_zero = input_map == 0
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)
    mask_to_flip = (zero_ranks <= num_to_pad.unsqueeze(1)) & is_zero
    output_map = torch.where(mask_to_flip, torch.ones_like(input_map), input_map)

    return output_map.transpose(0, 1)  # [num_tokens, num_experts]
