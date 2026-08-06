"""Fixed-contract int8 weight-only quantised matmul subsystem (in-scope, performance-critical).

``dq_matmul(a, b_q, scales)`` returns ``a @ dequant(b_q, scales)`` where ``a`` is
``[M, K]`` ``torch.float16`` on CUDA, ``b_q`` is ``[K, N]`` ``torch.int8`` (a
symmetric per-output-column quantised weight), and ``scales`` is ``[N]``
``torch.float16`` (one scale per output column). The dequantised weight is
``W[k, n] = b_q[k, n] * scales[n]``; the result is ``a @ W`` reduced with an fp32
accumulator and returned as fp16 ``[M, N]``. The benchmark drives a small-M,
large-K/N (weight-heavy, memory-bound) shape. The PUBLIC SIGNATURE and NUMERICAL
CONTRACT are fixed and MUST be preserved; only the implementation in this file is
in scope.

SLOW-BUT-CORRECT baseline: the whole ``[K, N]`` weight is DEQUANTISED into a full
dense floating-point buffer in global memory, and then a dense matmul is run on
that materialised buffer. The dequantised weight is written to and read back from
device memory (an extra pass), and it occupies far more bytes than the packed
int8 weight. The block is memory-bound; bring it up to production speed without
changing the contract.
"""
import torch


def dq_matmul(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b_q.ndim == 2 and scales.ndim == 1, "shape contract"
    assert a.dtype == torch.float16 and b_q.dtype == torch.int8, "fp16 a / int8 b_q"
    assert a.is_cuda and b_q.is_cuda and scales.is_cuda, "CUDA operands required"
    M, K = a.shape
    K2, N = b_q.shape
    assert K == K2 and scales.shape[0] == N, "inner/scale dim mismatch"
    # SLOW-BUT-CORRECT: materialise the full dequantised weight in fp32 global
    # memory, then a dense fp32 matmul. Extra [K, N] write + read; 4x the bytes of
    # the packed int8 weight; separate kernels.
    w = b_q.to(torch.float32) * scales.to(torch.float32).unsqueeze(0)
    c = torch.matmul(a.to(torch.float32), w)
    return c.to(torch.float16)
