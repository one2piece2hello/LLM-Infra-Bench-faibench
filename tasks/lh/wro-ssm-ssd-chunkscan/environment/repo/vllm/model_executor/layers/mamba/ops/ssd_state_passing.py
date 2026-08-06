# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.

# ruff: noqa: E501

# The chunk-boundary state recurrence is not provided in this baseline.
# The sequential reference in ssd_combined.py carries the running state
# directly across the whole sequence.

_NOT_AVAILABLE = (
    "chunk-boundary state passing is not implemented in this baseline; "
    "the sequential reference in ssd_combined.py is used instead")


def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None,
                       seq_idx=None, chunk_size=None, out_dtype=None,
                       is_cont_batched=False):
    raise NotImplementedError(_NOT_AVAILABLE)
