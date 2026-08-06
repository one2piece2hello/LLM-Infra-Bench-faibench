from __future__ import annotations

import torch
import torch.nn.functional as F


def gated_gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Apply a gated GELU activation over the last dimension.

    The input's last dimension is split in half. GELU is applied to the first
    half and multiplied by the second half. This straightforward implementation
    intentionally relies on ordinary PyTorch tensor operations.
    """
    if approximate not in ("none", "tanh"):
        raise ValueError("approximate must be 'none' or 'tanh'")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if x.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even")
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]
