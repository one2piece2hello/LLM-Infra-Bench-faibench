# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs

# ---------------------------------------------------------------------------
# cs321-llm-infra course task (STAGE-1 producer of the chunk-local WY/UT transform).
# The implementation of `chunk_scaled_dot_kkt_fwd` has been removed; implement it
# to the disclosed contract below. Its output `A` is consumed directly by
# `fla.ops.utils.solve_tril.solve_tril` (STAGE-2 consumer) to form the transform
# matrix T = (I + A)^{-1} used by the delta-rule / gated-delta chunk kernels.
# ---------------------------------------------------------------------------


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    For each chunk of `chunk_size` (``BT``) consecutive tokens and each head, build the
    strictly-lower-triangular matrix

        ``A[i, j] = beta[i] * (k[i] . k[j])``            for ``i > j`` (else 0)

    optionally scaled by the gate ratio ``2 ** (g[i] - g[j])`` when ``g`` is provided
    (``USE_G``). Rows are scaled by ``beta`` (a per-token, per-head scalar). The upper
    triangle and diagonal are zero. Tokens past the (possibly ragged / varlen) sequence
    end are masked to zero.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]` where `H` is the number of query/key heads.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, HV]` where `HV` is the number of value/output heads.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`
        chunk_indices (torch.LongTensor):
            The chunk indices of the input tensor. Default: None.
    Returns:
        beta * K * K^T of shape `[B, T, HV, BT]` where `BT` is the chunk size.
        For GVA, H < HV and HV % H == 0. For standard attention, H == HV.
    """
    raise NotImplementedError(
        "chunk_scaled_dot_kkt_fwd (STAGE-1 producer) is not implemented. "
        "Implement it to the contract in this docstring; its output feeds solve_tril.",
    )
