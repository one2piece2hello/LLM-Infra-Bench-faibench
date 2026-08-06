# Reviewer-only ORACLE (not baked into the image): the fast form. The discretization has no cross-timestep
# dependency, so every [Bt,L,D,N] entry is produced in ONE fused/vectorized pass via broadcasting:
# exp of the (delta x A) product over the whole tensor, and the (delta * B * u) triple broadcast.
# One handful of GPU kernels over the full tensor -- no per-timestep launches. Grounded in
# TRAIN.ARCH.SSM: johnma2006/mamba-minimal model.py:305 selective_scan (deltaA = exp(einsum('b l d,
# d n -> b l d n', delta, A)); deltaB_u = einsum('b l d, b l n, b l d -> b l d n', delta, B, u)).
# Never baked; used only to calibrate oracle_ms + prove headroom.
import torch


def discretize(u, delta, A, B):
    deltaA = torch.exp(delta.unsqueeze(-1) * A[None, None, :, :])            # [Bt,L,D,N]
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)        # [Bt,L,D,N]
    return deltaA, deltaB_u


def custom_kernel(data):
    u, delta, A, B = data
    return discretize(u, delta, A, B)
