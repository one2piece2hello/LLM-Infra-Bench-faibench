# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')


# Intra-chunk helpers for the gated delta-rule forward pass.
# The helpers below are part of this module's public surface inside the package;
# see the package's forward entry point for how they are used.


def chunk_gated_delta_rule_fwd_intra(*args, **kwargs):
    raise NotImplementedError(
        "chunk_gated_delta_rule_fwd_intra is not available in this build of "
        "fla.ops.gated_delta_rule.chunk_fwd")
