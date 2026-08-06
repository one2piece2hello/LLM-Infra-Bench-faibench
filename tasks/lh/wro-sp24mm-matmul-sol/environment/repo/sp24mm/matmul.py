"""Fixed-contract 2:4 semi-structured sparse weight / fp16-activation matmul subsystem
(in-scope, performance-critical).

``sp24mm_matmul(a, w_vals, w_meta)`` returns ``a @ W`` where the logical weight ``W`` is
``[K, N]`` fp16, 2:4 semi-structured sparse along K: within every group of 4 consecutive
K rows exactly TWO are nonzero. ``W`` is never stored densely -- it is COMPRESSED as

  * ``a``      is ``[M, K]`` ``torch.float16`` on CUDA (the activation),
  * ``w_vals`` is ``[K // 2, N]`` ``torch.float16`` on CUDA: the 2 nonzero weight values
    of each 4-row K-group, laid out in ascending K-order (group ``g`` occupies rows
    ``2*g`` and ``2*g + 1`` of ``w_vals``),
  * ``w_meta`` is ``[K // 4, N]`` ``torch.uint8`` on CUDA: per (group, column) it packs
    the two 2-bit indices (each in ``[0, 4)``, ascending) of the nonzero rows within the
    group -- the FIRST nonzero index in the low 2 bits, the SECOND in the next 2 bits.

The dense weight reconstructed from the compressed form is, for group ``g = k // 4`` and
in-group position ``r = k % 4``:

    i0 =  w_meta[g, n]        & 0x3          # index of the 1st nonzero row in the group
    i1 = (w_meta[g, n] >> 2)  & 0x3          # index of the 2nd nonzero row in the group
    W[4*g + i0, n] = w_vals[2*g,     n]
    W[4*g + i1, n] = w_vals[2*g + 1, n]
    W[4*g + r,  n] = 0                       for r not in {i0, i1}

and the result is ``a @ W`` reduced with an fp32 accumulator and returned as
``torch.float16`` ``[M, N]``. K is a multiple of 4. The benchmark drives a small-M,
large-K/N (weight-heavy) shape; the compressed weight is half the size of the dense
weight and only the 2 active activation columns per group contribute. The PUBLIC
SIGNATURE and NUMERICAL CONTRACT are fixed and MUST be preserved; only the
implementation in this file is in scope.

SLOW-BUT-CORRECT baseline: the compressed weight is first DECOMPRESSED into a full dense
``[K, N]`` fp16 buffer in global memory (scatter each of the 2 nonzeros per group to its
row via the metadata indices, zero elsewhere), and a dense matmul is then run on that
materialised buffer. The dense weight is written to and read back from device memory (an
extra full ``[K, N]`` pass), it is twice the bytes of the compressed representation, and
the matmul does the full dense K work including the structural zeros. The block is
memory/compute-bound; bring it up to production speed without changing the contract.
"""
import torch


def sp24mm_matmul(a: torch.Tensor, w_vals: torch.Tensor,
                  w_meta: torch.Tensor) -> torch.Tensor:
    """Return ``a @ W`` as fp16 ``[M, N]`` for a 2:4 semi-structured sparse ``W``.

    Contract (all tensors CUDA):
      a      : fp16  [M, K]                 (K a multiple of 4)
      w_vals : fp16  [K // 2, N]            (2 nonzeros per 4-row K-group, K-order)
      w_meta : uint8 [K // 4, N]            (two 2-bit in-group nonzero indices per group:
                                             low 2 bits = 1st, next 2 bits = 2nd)
    """
    M, K = a.shape
    Kh, N = w_vals.shape
    Kg = w_meta.shape[0]
    assert K % 4 == 0, "K must be a multiple of 4"
    assert Kh == K // 2, "w_vals must have K//2 rows"
    assert Kg == K // 4 and w_meta.shape[1] == N, "w_meta shape mismatch"

    # SLOW-BUT-CORRECT: decompress the 2:4 weight into a full dense [K, N] fp16 buffer in
    # global memory (scatter the 2 nonzeros of each group to their metadata rows, zeros
    # elsewhere), then run a dense fp32 matmul on it. An extra full [K, N] write/read for
    # the densified weight (twice the bytes of the compressed form), and the matmul does
    # the full dense K work including the structural zeros.
    Kg = K // 4
    meta = w_meta.to(torch.int64)                                # [Kg, N]
    i0 = (meta & 0x3)                                            # [Kg, N] 1st nonzero row
    i1 = ((meta >> 2) & 0x3)                                     # [Kg, N] 2nd nonzero row
    v0 = w_vals[0::2, :].to(torch.float32)                       # [Kg, N] 1st nonzero value
    v1 = w_vals[1::2, :].to(torch.float32)                       # [Kg, N] 2nd nonzero value

    w = torch.zeros((Kg, 4, N), dtype=torch.float32, device=a.device)  # [Kg, 4, N] dense group
    nidx = torch.arange(N, device=a.device)[None, :].expand(Kg, N)     # [Kg, N]
    gidx = torch.arange(Kg, device=a.device)[:, None].expand(Kg, N)    # [Kg, N]
    w[gidx, i0, nidx] = v0
    w[gidx, i1, nidx] = v1
    w = w.reshape(K, N)                                          # [K, N] dense fp32 weight
    c = torch.matmul(a.to(torch.float32), w)                     # [M, N] fp32
    return c.to(torch.float16)
