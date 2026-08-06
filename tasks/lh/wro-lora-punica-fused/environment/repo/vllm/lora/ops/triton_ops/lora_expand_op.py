# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

import torch

from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_b_ptr
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op



def _lora_expand_kernel(*args, **kwargs):
    raise NotImplementedError("fused expand kernel removed in this baseline")


def _lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    # Straightforward expand: for each (slice, token) with an assigned LoRA id,
    # project the shrunk buffer through that LoRA's B matrix and write it into
    # the slice's region of the output (adding to what is already there when
    # add_inputs). inputs is [num_slices, num_tokens, rank]; output_tensor is
    # [num_tokens, sum(out_s)] and is written in place.
    if bool(no_lora_flag_cpu.item()):
        return
    num_slices = len(lora_b_weights)
    tlm = token_lora_mapping
    active = [int(v) for v in torch.unique(tlm).tolist() if int(v) >= 0]
    off = offset_start
    for s in range(num_slices):
        B = lora_b_weights[s].to(torch.float32)
        out_s = B.shape[-2]
        rank = B.shape[-1]
        buf = inputs[s].to(torch.float32)
        for lid in active:
            mask = tlm == lid
            if not bool(mask.any()):
                continue
            Bw = B[lid].reshape(out_s, rank)
            delta = buf[mask] @ Bw.t()
            region = output_tensor[mask, off:off + out_s]
            if add_inputs:
                output_tensor[mask, off:off + out_s] = (region.to(torch.float32) + delta).to(output_tensor.dtype)
            else:
                output_tensor[mask, off:off + out_s] = delta.to(output_tensor.dtype)
        off += out_s


def _lora_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_expand",
        op_func=_lora_expand,
        mutates_args=["output_tensor"],
        fake_impl=_lora_expand_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    lora_expand = torch.ops.vllm.lora_expand

except AttributeError:
    lora_expand = _lora_expand
