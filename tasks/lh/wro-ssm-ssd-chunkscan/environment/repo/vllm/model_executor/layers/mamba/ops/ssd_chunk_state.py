# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.

# ruff: noqa: E501

# The chunked per-block state computation is not provided in this baseline.
# The sequential reference in ssd_combined.py computes the SSM state directly.

_NOT_AVAILABLE = (
    "chunked per-block state computation is not implemented in this baseline; "
    "the sequential reference in ssd_combined.py is used instead")


def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False,
                      dt_limit=(0.0, float("inf"))):
    raise NotImplementedError(_NOT_AVAILABLE)


def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None,
                     states_in_fp32=True):
    raise NotImplementedError(_NOT_AVAILABLE)


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states,
                       initial_states=None):
    raise NotImplementedError(_NOT_AVAILABLE)
