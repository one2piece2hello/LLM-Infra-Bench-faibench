# Performance Optimization Task — submission entry point.
#
# Implement `discretize` to the contract in instruction.md, then make it as fast as possible.
# This is the ONLY file you edit. The verifier grades numerical correctness first (fp32 output
# within rtol=atol=1e-3 of a seeded reference), then latency on GPU (H20). A submission that
# leaves NotImplementedError in place scores 0.

import torch


def discretize(u, delta, A, B):
    """Zero-order-hold (ZOH) discretization of a selective state-space model's parameters — the
    per-timestep preprocessing that precedes the selective scan.

    Args:
        u:     float32 CUDA tensor, shape [Bt, L, D]  (batch, sequence length, inner channels).
        delta: float32 CUDA tensor, shape [Bt, L, D]  (per-position, per-channel step; positive).
        A:     float32 CUDA tensor, shape [D, N]      (state matrix, one N-dim state per channel).
        B:     float32 CUDA tensor, shape [Bt, L, N]  (input projection; input-dependent).

    Produce two tensors, each of shape [Bt, L, D, N]:

        deltaA[b, l, d, n]   = exp( delta[b, l, d] * A[d, n] )           # ZOH on A (note the exp)
        deltaB_u[b, l, d, n] = delta[b, l, d] * B[b, l, n] * u[b, l, d]  # (Euler on B) * input u

    There is no cross-timestep dependency — every [b,l,d,n] entry is an independent function of the
    inputs. All arithmetic is float32.

    Returns:
        (deltaA, deltaB_u): a tuple of two float32 tensors, each shape [Bt, L, D, N].
    """
    raise NotImplementedError("implement discretize to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    data = (u, delta, A, B). Already wired to call discretize and return the (deltaA, deltaB_u)
    tuple.
    """
    u, delta, A, B = data
    return discretize(u, delta, A, B)
