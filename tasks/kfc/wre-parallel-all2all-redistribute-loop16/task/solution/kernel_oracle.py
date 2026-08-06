# Reviewer-only ORACLE: the FAST vectorized redistribution. Never baked into any
# image. Used only to (a) calibrate the oracle_ms latency constant on the authoring lane and (b) prove the
# correctness + headroom gradient. vs_oracle=1.0 anchor.
#
# Mechanism (the deliberately-scrubbed technique; the contract states only the y[d,s]=x[s,d] block
# transpose): the whole redistribution is a SINGLE swap of the two leading rank axes followed by ONE
# contiguous materialization -- torch's transpose is a metadata view (zero data movement) and
# .contiguous() is ONE coalesced copy kernel over the entire buffer. The host issues O(1) kernels
# regardless of world_size, versus the naive per-(src,dst)-block copy loop that issues O(W^2)
# separate small copies. The launch-overhead reduction is what is measured single-GPU and it grows
# with world_size. This mirrors the all-to-all dispatch/combine reorg in expert/sequence parallelism.
import torch


def all_to_all_redistribute(x, world_size):
    # swap source<->destination rank axes (metadata-only view) + one contiguous copy of the whole buffer
    return x.transpose(0, 1).contiguous()


def custom_kernel(data):
    x, config = data
    return all_to_all_redistribute(x, config["world_size"])
