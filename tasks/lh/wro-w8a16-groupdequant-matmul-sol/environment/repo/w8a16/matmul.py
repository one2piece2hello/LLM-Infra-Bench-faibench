"""Fixed-contract group-quantised int8 weight / fp16 activation matmul subsystem
(in-scope, performance-critical).

``w8a16_matmul(a, qweight, scales, zeros, group_size)`` returns
``a @ dequant(qweight, scales, zeros, group_size)`` where

  * ``a``       is ``[M, K]`` ``torch.float16`` on CUDA (the activation),
  * ``qweight`` is ``[K, N]`` ``torch.int8`` on CUDA (an ASYMMETRIC group-quantised
    weight: consecutive blocks of ``group_size`` rows along K share one
    ``(scale, zero)`` pair per output column),
  * ``scales``  is ``[K // group_size, N]`` ``torch.float16`` on CUDA (one scale per
    (group, column)),
  * ``zeros``   is ``[K // group_size, N]`` ``torch.int8`` on CUDA (one integer
    zero-point per (group, column)),
  * ``group_size`` is a positive int dividing ``K``.

The dequantised weight is

    W[k, n] = (qweight[k, n] - zeros[k // group_size, n]) * scales[k // group_size, n]

and the result is ``a @ W`` reduced with an fp32 accumulator and returned as
``torch.float16`` ``[M, N]``. The benchmark drives a small-M, large-K/N
(weight-heavy, memory-bound) shape. The PUBLIC SIGNATURE and NUMERICAL CONTRACT are
fixed and MUST be preserved; only the implementation in this file is in scope.

SLOW-BUT-CORRECT baseline: the per-group ``scales`` and ``zeros`` are first EXPANDED
back to the full ``[K, N]`` grid (one entry per weight element), the whole ``[K, N]``
weight is then DEQUANTISED into a full dense floating-point buffer in global memory,
and finally a dense matmul is run on that materialised buffer. The expanded scale and
zero grids and the dequantised weight are all written to and read back from device
memory (several extra passes), and together they occupy far more bytes than the packed
int8 weight. The block is memory-bound; bring it up to production speed without
changing the contract.
"""
import torch


def w8a16_matmul(a: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                 zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    """Return ``a @ dequant(qweight, scales, zeros, group_size)`` as fp16 ``[M, N]``.

    Contract (all tensors CUDA):
      a       : fp16 [M, K]
      qweight : int8 [K, N]              (asymmetric group-quantised weight)
      scales  : fp16 [K // group_size, N]
      zeros   : int8 [K // group_size, N]
      group_size : int, divides K
    """
    M, K = a.shape
    K2, N = qweight.shape
    G = K // group_size
    assert K == K2, "inner dim mismatch"
    assert scales.shape[0] == G and scales.shape[1] == N, "scales shape mismatch"
    assert zeros.shape[0] == G and zeros.shape[1] == N, "zeros shape mismatch"
    assert K % group_size == 0, "K must be a multiple of group_size"

    # SLOW-BUT-CORRECT: expand the per-group (scale, zero) grids to the full [K, N]
    # resolution, materialise the full dequantised weight in fp32 global memory, and
    # run a dense fp32 matmul on it. Extra [K, N] writes/reads for the expanded scale
    # grid, the expanded zero grid, and the dense weight; far more bytes than the
    # packed int8 weight; separate kernels.
    scales_full = scales.to(torch.float32).repeat_interleave(group_size, dim=0)  # [K, N]
    zeros_full = zeros.to(torch.float32).repeat_interleave(group_size, dim=0)     # [K, N]
    w = (qweight.to(torch.float32) - zeros_full) * scales_full                    # [K, N] fp32
    c = torch.matmul(a.to(torch.float32), w)                                      # [M, N] fp32
    return c.to(torch.float16)
