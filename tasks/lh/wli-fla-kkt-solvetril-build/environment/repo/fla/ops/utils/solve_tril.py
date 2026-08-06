# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_TMA_SUPPORTED, autotune_cache_kwargs, input_guard

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
assert FLA_TRIL_PRECISION in ['ieee', 'tf32', 'tf32x3'], \
    f"FLA_TRIL_PRECISION must be one of 'ieee', 'tf32', or 'tf32x3', but got {FLA_TRIL_PRECISION}"
DOT_PRECISION_AUTOTUNE_LIST = ["ieee"] if not IS_TMA_SUPPORTED else list({"ieee", FLA_TRIL_PRECISION})

# ---------------------------------------------------------------------------
# cs321-llm-infra course task (STAGE-2 consumer of the chunk-local WY/UT transform).
# The implementation of `solve_tril` has been removed; implement it to the disclosed
# contract below. Its input `A` is exactly the output of
# `fla.ops.common.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd` (STAGE-1 producer):
# a per-chunk strictly-lower-triangular matrix. `solve_tril` returns the per-chunk
# inverse (I + A)^{-1}, i.e. the transform matrix T used by delta-rule chunk kernels.
# A correct, memory-efficient implementation tiles each BT x BT chunk into 16 x 16
# sub-blocks (invert the diagonal sub-blocks, then combine the off-diagonal sub-blocks).
# ---------------------------------------------------------------------------


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    The inverse is computed independently per chunk block along the sequence: for each
    contiguous ``BT`` x ``BT`` block on the (block-)diagonal, return ``(I + A_block)^{-1}``.
    Because ``I + A_block`` is unit lower-triangular, the result is unit lower-triangular
    with the same shape as the input. ``BT`` (``A.shape[-1]``) is only ever 16, 32, or 64.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    raise NotImplementedError(
        "solve_tril (STAGE-2 consumer) is not implemented. Implement it to the contract "
        "in this docstring; its input A is the output of chunk_scaled_dot_kkt_fwd.",
    )
