# Reviewer-only NEGATIVE (not baked into the image): fast but WRONG. Uses a CIRCULAR convolution (FFT at
# length L with no zero-padding) instead of the causal linear convolution. The missing zero-pad
# lets the tail of the sequence wrap around into early outputs, so a position t reads "future"
# samples via the wrap -- it violates causality and disagrees with the reference for any L > 1.
# It is a single fast FFT pass, so it must FAIL the correctness gate (not the timing).
import torch


def causal_conv(u, k):
    B, H, L = u.shape
    uf = torch.fft.rfft(u.float(), n=L, dim=-1)             # WRONG: n=L -> circular, no padding
    kf = torch.fft.rfft(k.float(), n=L, dim=-1).unsqueeze(0)
    y = torch.fft.irfft(uf * kf, n=L, dim=-1)[..., :L]
    return y.to(torch.bfloat16)


def custom_kernel(data):
    u, k, config = data
    return causal_conv(u, k)
