# Performance Optimization Task — submission entry point.
#
# Implement `reduce_partials` to the contract in instruction.md, then make it as fast as
# possible. This is the ONLY file you edit. The verifier grades numerical correctness first
# (bf16 output within rtol=atol=2e-2 of a seeded fp32 reference), then latency on GPU (H20).
# A submission that leaves NotImplementedError in place scores 0.

import torch


def reduce_partials(partials, bias):
    """Combine the per-rank partial outputs of a tensor-parallel row-parallel linear.

    In tensor-parallel serving, a row-parallel linear splits its input dimension across `R` ranks;
    each rank produces a partial output over the SAME [T, D] output shape, and the final result is
    the sum of those partials across ranks with the bias added exactly ONCE (the "all-reduce +
    bias" epilogue). You implement that combine.

    Args:
        partials: bfloat16 CUDA tensor, shape [R, T, D] — ``partials[r]`` is rank ``r``'s partial
                  output ([T, D]). ``R`` is the tensor-parallel world size.
        bias:     bfloat16 CUDA tensor, shape [D] — added once to the reduced result (broadcast
                  over the T rows).

    Compute (accumulate in fp32 for numerical stability, then cast back to bf16):

        acc = sum over r in [0, R) of partials[r]      # reduce over the rank axis, fp32 [T, D]
        out = acc + bias                                # bias added ONCE (broadcast over rows), fp32
        return out.to(bfloat16)                         # [T, D]

    Returns:
        out: bfloat16 tensor, shape [T, D].
    """
    raise NotImplementedError("implement reduce_partials to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    data = (partials, bias, config) where config = {"R": int, "T": int, "D": int}. Already wired to
    call reduce_partials and return the reduced [T, D] output.
    """
    partials, bias, config = data
    return reduce_partials(partials, bias)
