"""sp24mm — a 2:4 semi-structured sparse weight / fp16-activation matmul subsystem.

Public contract (fixed; the verifier imports exactly this):

    sp24mm_matmul(a, w_vals, w_meta) -> torch.Tensor

Computes ``a @ W`` where ``W`` (logical shape ``[K, N]``, fp16) is 2:4 semi-structured
sparse along K: within every group of 4 consecutive K rows exactly 2 are nonzero. The
weight is stored COMPRESSED as ``w_vals`` ``[K // 2, N]`` (the 2 nonzero values per
group, in K-order) plus ``w_meta`` ``[K // 4, N]`` uint8 holding the two 2-bit indices
(within ``[0, 4)``) of the nonzero rows per group. See ``matmul.py`` for the exact
layout and numerical contract.
"""
from .matmul import sp24mm_matmul

__all__ = ["sp24mm_matmul"]
