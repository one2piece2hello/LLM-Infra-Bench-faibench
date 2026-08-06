# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

import torch

from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op



def _lora_shrink_kernel(*args, **kwargs):
    raise NotImplementedError("fused shrink kernel removed in this baseline")


def _lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    scaling: float,
) -> None:
    # Straightforward shrink: for each (slice, token) with an assigned LoRA id,
    # project the token through that LoRA's A matrix and scale. output_tensor is
    # [num_slices, num_tokens, rank] and is written in place. Tokens are grouped
    # by LoRA id one id at a time.
    if bool(no_lora_flag_cpu.item()):
        return
    num_slices = len(lora_a_weights)
    rank = lora_a_weights[0].shape[-2]
    hidden = lora_a_weights[0].shape[-1]
    x = inputs.to(torch.float32)
    tlm = token_lora_mapping
    active = [int(v) for v in torch.unique(tlm).tolist() if int(v) >= 0]
    for s in range(num_slices):
        A = lora_a_weights[s].to(torch.float32)
        for lid in active:
            mask = tlm == lid
            if not bool(mask.any()):
                continue
            Aw = A[lid].reshape(rank, hidden)
            out_sl = (x[mask] @ Aw.t()) * scaling
            output_tensor[s][mask] = out_sl.to(output_tensor.dtype)


def _lora_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    scaling: float,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_shrink",
        op_func=_lora_shrink,
        mutates_args=["output_tensor"],
        fake_impl=_lora_shrink_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    lora_shrink = torch.ops.vllm.lora_shrink

except AttributeError:
    lora_shrink = _lora_shrink
