# Native sparse attention (NSA) forward pass for this repository.
# Public entry point: `parallel_nsa`. Its signature defines the accepted tensor
# shapes, the GQA relationship between query and key/value heads, the block
# selection inputs and the optional variable-length (`cu_seqlens`) form.
# The observable output contract is fixed by the repository's tests.
import torch

from fla.ops.nsa.naive import naive_nsa


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
) -> torch.Tensor:
    out = naive_nsa(
        q, k, v,
        g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.to(q.dtype)
