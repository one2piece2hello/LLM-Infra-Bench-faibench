# Reviewer-only NEGATIVE (not baked into the image): fast but WRONG. Adds the bias to EVERY rank's partial
# before the reduction, so the bias is summed R times instead of once. The row-parallel "all-reduce
# + bias" epilogue requires the bias added exactly once (nano-vllm/TGI add it only on rank 0); this
# counts it R times, so for R > 1 (and any non-zero bias) the result is off by (R-1)*bias and it
# FAILS the correctness gate. It is still a single fast fused pass -- fast but wrong.
import torch


def reduce_partials(partials, bias):
    R, T, D = partials.shape
    biased = partials.float() + bias.float()          # BUG: bias added to all R ranks
    acc = biased.sum(dim=0)                            # -> bias counted R times
    return acc.to(torch.bfloat16)


def custom_kernel(data):
    partials, bias, config = data
    return reduce_partials(partials, bias)
