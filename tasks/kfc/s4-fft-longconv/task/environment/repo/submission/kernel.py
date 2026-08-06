# Performance Optimization Task — submission entry point.
#
# Implement `causal_conv` to the contract in instruction.md, then make it as fast as possible.
# This is the ONLY file you edit. The verifier grades numerical correctness first (bf16 output
# within rtol=atol=2e-2 of a seeded fp32 reference), then latency on GPU (H20). A submission that
# leaves NotImplementedError in place scores 0.

import torch


def causal_conv(u, k):
    """Causal (non-circular) 1-D convolution of each input sequence with its channel's kernel.

    Args:
        u: bfloat16 tensor, shape [B, H, L] (B sequences, H channels, length L).
        k: bfloat16 tensor, shape [H, L] (per-channel causal kernel, full length L).

    For every (b, h) and output position t in [0, L):

        y[b, h, t] = sum_{s=0}^{t} k[h, t - s] * u[b, h, s]

    i.e. the linear convolution of u[b,h] with k[h], truncated to the first L samples (a causal
    system: output t depends only on inputs s <= t; no wrap-around from the tail of the sequence).
    Compute the accumulation in fp32, then cast the result back to bf16.

    Returns:
        bfloat16 tensor, shape [B, H, L].
    """
    raise NotImplementedError("implement causal_conv to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    data = (u, k, config) where config = {"L": int}. Already wired to call causal_conv and return
    the convolved sequences.
    """
    u, k, config = data
    return causal_conv(u, k)
