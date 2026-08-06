# Reviewer-only NEGATIVE known-bad (not baked into the image): FAST but WRONG. Must FAIL the correctness gate.
# It uses the fast path (transpose + contiguous, so latency is NOT the thing that stops it) but
# transposes the WRONG pair of axes: it swaps the destination-rank axis with the `chunk` axis
# (dims 1 and 2) instead of swapping the two rank axes (dims 0 and 1). The payload lands in the
# wrong block positions, so y[d, s] != x[s, d] and the fp32-reference check rejects it -> 0.
# (Also, for chunk != world_size the shape itself comes out wrong, an immediate fail.)
import torch


def all_to_all_redistribute(x, world_size):
    return x.transpose(1, 2).contiguous()         # WRONG axes: dst<->chunk instead of src<->dst


def custom_kernel(data):
    x, config = data
    return all_to_all_redistribute(x, config["world_size"])
