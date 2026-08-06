# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.

# ruff: noqa: E501

# The chunked output computation (diagonal + off-diagonal blocks) is not
# provided in this baseline. The sequential reference in ssd_combined.py reads
# the output out of the running state at each time step.

_NOT_AVAILABLE = (
    "chunked scan output is not implemented in this baseline; the sequential "
    "reference in ssd_combined.py is used instead")


def _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=None, z=None,
                    seq_idx=None, chunk_indices=None, chunk_offsets=None,
                    initial_states=None, out=None):
    raise NotImplementedError(_NOT_AVAILABLE)
