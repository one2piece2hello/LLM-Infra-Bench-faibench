# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.

# ruff: noqa: E501

# The batched C^T B chunk matmul is not provided in this baseline; the
# sequential reference in ssd_combined.py forms the C . state read-out
# directly and needs no precomputed CB blocks.

_NOT_AVAILABLE = (
    "chunked batched matmul (CB) is not implemented in this baseline; the "
    "sequential reference in ssd_combined.py is used instead")


def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False,
                   output_dtype=None):
    raise NotImplementedError(_NOT_AVAILABLE)
