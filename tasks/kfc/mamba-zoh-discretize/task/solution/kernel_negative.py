# Reviewer-only NEGATIVE (not baked into the image): fast but WRONG discretization. Drops the exponential on the
# state-transition factor -- uses deltaA = delta * A (a linear/Euler step) instead of the ZOH
# deltaA = exp(delta * A). Since A is negative and delta positive, the correct factor lies in (0,1)
# while this wrong one is negative -- a completely different value. It is a single fast fused pass,
# so it must FAIL the correctness gate (not the timing). deltaB_u is left correct, so only the
# disclosed ZOH invariant is broken.
import torch


def discretize(u, delta, A, B):
    deltaA = delta.unsqueeze(-1) * A[None, None, :, :]                       # WRONG: no exp (Euler, not ZOH)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    return deltaA, deltaB_u


def custom_kernel(data):
    u, delta, A, B = data
    return discretize(u, delta, A, B)
