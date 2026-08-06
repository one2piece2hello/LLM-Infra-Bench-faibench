"""blocksp — a structured K-block-sparse weight / fp16-activation matmul subsystem.

Public contract (fixed; the verifier imports exactly this):

    blocksp_matmul(a, w_blocks, k_idx, block_k) -> torch.Tensor

Computes ``a @ W`` where the logical weight ``W`` (``[K, N]``, fp16) is STRUCTURED
block-sparse along K: K is partitioned into ``K // block_k`` contiguous row-blocks and
only a subset are nonzero (structured input-feature-block pruning). ``W`` is stored
COMPRESSED as ``w_blocks`` ``[nnz * block_k, N]`` (the nonzero row-blocks stacked in
ascending block order) plus ``k_idx`` ``[nnz]`` int32 (the logical block index of each
stored block). See ``matmul.py`` for the exact layout and numerical contract.
"""
from .matmul import blocksp_matmul

__all__ = ["blocksp_matmul"]
