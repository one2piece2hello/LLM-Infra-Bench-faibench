"""Fixed-contract structured K-block-sparse weight / fp16-activation matmul subsystem
(in-scope, performance-critical).

``blocksp_matmul(a, w_blocks, k_idx, block_k)`` returns ``a @ W`` where the logical
weight ``W`` is ``[K, N]`` fp16, STRUCTURED block-sparse along K: the ``K`` rows are
partitioned into ``num_blocks = K // block_k`` contiguous row-blocks and only ``nnz`` of
them are nonzero (structured input-feature-block pruning; the same blocks are pruned for
every output column). ``W`` is never stored densely -- it is COMPRESSED as

  * ``a``        is ``[M, K]`` ``torch.float16`` on CUDA (the activation),
  * ``w_blocks`` is ``[nnz * block_k, N]`` ``torch.float16`` on CUDA: the nonzero
    row-blocks stacked in ASCENDING block order (stored block ``p`` occupies rows
    ``p*block_k : (p+1)*block_k``),
  * ``k_idx``    is ``[nnz]`` ``torch.int32`` on CUDA: the logical block index (in
    ``[0, num_blocks)``, ascending, distinct) of each stored block,
  * ``block_k``  is a positive int dividing ``K``.

The dense weight reconstructed from the compressed form is, for logical block
``kb = k // block_k`` and in-block row ``r = k % block_k``:

    W[k, n] = w_blocks[p * block_k + r, n]   if kb == k_idx[p] for some stored p
            = 0                              if kb is not in k_idx  (a pruned block)

and the result is ``a @ W`` reduced with an fp32 accumulator and returned as
``torch.float16`` ``[M, N]``. The benchmark drives a small-M, large-K/N (weight-heavy)
shape with a LOW block-density (few nonzero blocks): only the ``nnz`` stored blocks
contribute, so both the weight traffic and the matmul work scale with ``nnz``, not the
full ``K``. The PUBLIC SIGNATURE and NUMERICAL CONTRACT are fixed and MUST be preserved;
only the implementation in this file is in scope.

SLOW-BUT-CORRECT baseline: the compressed blocks are first SCATTERED back into a full
dense ``[K, N]`` fp16 weight buffer in global memory (zeros for every pruned block), and
a dense matmul is then run on that materialised buffer. The dense weight is written to
and read back from device memory (a full ``[K, N]`` pass, ``num_blocks / nnz`` times the
bytes of the stored blocks), and the matmul does the full dense ``K`` work including the
pruned (zero) blocks. The block is memory/compute-bound; bring it up to production speed
without changing the contract.
"""
import torch


def blocksp_matmul(a: torch.Tensor, w_blocks: torch.Tensor, k_idx: torch.Tensor,
                   block_k: int) -> torch.Tensor:
    """Return ``a @ W`` as fp16 ``[M, N]`` for a structured K-block-sparse ``W``.

    Contract (all tensors CUDA):
      a        : fp16  [M, K]                 (K a multiple of block_k)
      w_blocks : fp16  [nnz * block_k, N]     (nonzero K row-blocks, ascending block order)
      k_idx    : int32 [nnz]                  (logical block index of each stored block)
      block_k  : int, divides K
    """
    M, K = a.shape
    KB, N = w_blocks.shape
    nnz = k_idx.shape[0]
    assert K % block_k == 0, "block_k must divide K"
    assert KB == nnz * block_k, "w_blocks must have nnz*block_k rows"

    # SLOW-BUT-CORRECT: scatter the stored nonzero blocks back into a full dense [K, N]
    # fp16 weight (zeros for pruned blocks), then run a dense fp32 matmul on it. An extra
    # full [K, N] write/read for the densified weight (num_blocks/nnz times the bytes of
    # the stored blocks), and the matmul does the full dense K work including zero blocks.
    w = torch.zeros((K, N), dtype=torch.float32, device=a.device)
    kidx = k_idx.to(torch.int64)
    for p in range(nnz):
        kb = int(kidx[p].item())
        w[kb * block_k:(kb + 1) * block_k, :] = w_blocks[p * block_k:(p + 1) * block_k, :].to(torch.float32)
    c = torch.matmul(a.to(torch.float32), w)                 # [M, N] fp32
    return c.to(torch.float16)
