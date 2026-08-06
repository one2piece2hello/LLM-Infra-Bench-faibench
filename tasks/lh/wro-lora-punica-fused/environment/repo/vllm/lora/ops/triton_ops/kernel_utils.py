# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for Punica kernel construction.
"""
from vllm.triton_utils import tl, triton



def mm_k(*args, **kwargs):
    raise NotImplementedError("tiled matmul kernel removed in this baseline")


def do_expand_kernel(*args, **kwargs):
    raise NotImplementedError("fused expand kernel removed in this baseline")


def do_shrink_kernel(*args, **kwargs):
    raise NotImplementedError("fused shrink kernel removed in this baseline")
