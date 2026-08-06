# Reviewer-only ORACLE (not baked into the image): the fast form. A causal 1-D convolution with a full-length
# kernel is evaluated in O(L log L) via the FFT: zero-pad both operands to length >= 2L-1, take
# the real FFT, multiply in the frequency domain, inverse-FFT, and keep the first L samples (the
# zero-padding makes the circular convolution equal the causal linear convolution). Grounded in
# TRAIN.ARCH.SSM: state-spaces/s4 — the S4/S4D "convolution mode" materializes the SSM kernel of
# length L and applies it to the sequence with an FFT (the global long convolution), instead of
# stepping the O(L^2) recurrence/direct sum. Never baked; calibrates oracle_ms + proves headroom.
import torch


def causal_conv(u, k):
    B, H, L = u.shape
    n = 1
    while n < 2 * L:
        n <<= 1
    uf = torch.fft.rfft(u.float(), n=n, dim=-1)
    kf = torch.fft.rfft(k.float(), n=n, dim=-1).unsqueeze(0)  # [1,H,.]
    y = torch.fft.irfft(uf * kf, n=n, dim=-1)[..., :L]
    return y.to(torch.bfloat16)


def custom_kernel(data):
    u, k, config = data
    return causal_conv(u, k)
