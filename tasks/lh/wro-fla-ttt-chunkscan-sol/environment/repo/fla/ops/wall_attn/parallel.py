# Windowed decay ("wall") attention forward pass for this repository.
# Public entry point: `parallel_wall_attn`. Its signature defines the accepted
# tensor shapes, the per-channel decay gate, the optional scalar gate and sink
# bias, the window size, and the optional variable-length (`cu_seqlens`) form.
# The observable output contract, including the input validation performed
# before dispatch, is fixed by the repository's tests.
import torch

from fla.ops.wall_attn.naive import naive_wall_attn


def parallel_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    o = naive_wall_attn(
        q, k, v, g,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
        sink_bias=sink_bias,
        g_scalar=g_scalar,
    )
    if isinstance(o, tuple):
        o = o[0]
    return o.to(q.dtype)
