# Performance Optimization Task — submission entry point.
#
# Implement the all-to-all redistribution routine below to the contract in
# instruction.md, then make it as fast as possible. This is the ONLY file you edit.
#
# The verifier drives `custom_kernel` on hidden block-partitioned bf16 CUDA tensors
# and grades numerical correctness first (rtol=atol=2e-2 vs a seeded fp32 reference),
# then latency. A submission that leaves NotImplementedError in place scores 0.

import torch


def all_to_all_redistribute(x, world_size):
    """All-to-all block redistribution across a simulated `world_size` participant grid.

    In expert / sequence parallelism, an all-to-all takes a buffer partitioned by the
    DESTINATION rank of each block and returns it partitioned by the SOURCE rank — every
    rank `i` ends up holding the blocks that each rank `j` addressed to it. Simulated
    single-GPU, this is a block transpose of the two leading (rank) axes.

    Args:
        x:          bfloat16 CUDA tensor, shape [world_size, world_size, chunk, D].
                    `x[s, d]` is the `[chunk, D]` block that source rank `s` sends to
                    destination rank `d`.
        world_size: int W (== x.shape[0] == x.shape[1]).

    Semantics:
        Return `y`, a **contiguous** bfloat16 tensor of shape [world_size, world_size, chunk, D]
        with

            y[d, s] = x[s, d]        for all s, d in [0, world_size)

        i.e. swap the two leading rank axes (source<->destination) and materialize the result
        contiguously. The `[chunk, D]` payload of each block is copied unchanged; only its
        (source, destination) position moves. dtype and shape are preserved.

    Return `y` as a contiguous bfloat16 tensor of shape [world_size, world_size, chunk, D].
    """
    raise NotImplementedError("implement all_to_all_redistribute to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    Args:
        data = (x, config) where
            x:      bfloat16 CUDA [world_size, world_size, chunk, D].
            config: {"world_size": int, "chunk": int, "D": int}.

    Returns:
        y — the redistributed contiguous bfloat16 tensor (see all_to_all_redistribute above).
    """
    x, config = data
    return all_to_all_redistribute(x, config["world_size"])
