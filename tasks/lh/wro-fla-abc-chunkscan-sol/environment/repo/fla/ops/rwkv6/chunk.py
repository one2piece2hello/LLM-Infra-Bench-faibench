# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

__all__ = ['chunk_rwkv6', 'chunk_rwkv6_fwd_cumsum']


def chunk_rwkv6_fwd_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Within-chunk inclusive / exclusive cumulative sum of the log-decay `g`.

    Returns ``(gi, ge)`` both fp32 with the same ``[B, T, H, S]`` shape as ``g``:
    ``gi`` is the inclusive (<=) prefix sum over each length-``chunk_size`` block
    along T, ``ge`` the exclusive (<) one. Kept in fp32; optionally pre-scaled.
    A plain-torch equivalent of the reference block-cumsum (correct, not tiled).
    """
    if cu_seqlens is not None:
        raise NotImplementedError("variable-length (cu_seqlens) cumsum is out of scope")
    B, T, H, S = g.shape
    BT = chunk_size
    gf = g.float()
    NT = (T + BT - 1) // BT
    pad = NT * BT - T
    if pad:
        gf = torch.nn.functional.pad(gf, (0, 0, 0, 0, 0, pad))
    gf = gf.view(B, NT, BT, H, S)
    gi = gf.cumsum(dim=2)
    ge = gi - gf
    gi = gi.reshape(B, NT * BT, H, S)[:, :T]
    ge = ge.reshape(B, NT * BT, H, S)[:, :T]
    if scale is not None:
        gi = gi * scale
        ge = ge * scale
    return gi.contiguous(), ge.contiguous()


@torch.compiler.disable
def chunk_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    RWKV-6 (Finch) data-dependent linear-attention forward.

    Args:
        r (torch.Tensor): receptance/queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        w (torch.Tensor): log-space forget gates of shape `[B, T, H, K]` applied to keys
            (per channel; `exp(w) in (0, 1]` keeps the recurrence contractive).
        u (torch.Tensor): first-token bonus of shape `[H, K]`.
        scale (Optional[float]): score scale; defaults to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]): initial state `[N, H, K, V]`.
        output_final_state (bool): whether to also return the final `[N, H, K, V]` state.
        cu_seqlens (Optional[torch.LongTensor]): variable-length cumulative sequence lengths.

    Returns:
        o (torch.Tensor): outputs of shape `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]): final state `[N, H, K, V]` or `None`.

    ---------------------------------------------------------------------------
    SLOW-BUT-CORRECT forward: an idiomatic eager per-time-step RWKV-6 recurrence
    (the repo's own ``naive_recurrent_rwkv6`` semantics, re-expressed in the public
    ``[B, T, H, *]`` layout), written in plain torch so it is numerically correct
    and autograd-differentiable but evaluates the state scan strictly one dependent
    time step at a time over T -> O(T) sequential elementwise launches, no time
    parallelism. Preserves the public (o, final_state) return contract, the per-
    channel data-dependent decay ``exp(w)``, and the current-token bonus ``u``.
    Making this forward's run time stop growing linearly with the number of
    dependent time steps is the task.
    ---------------------------------------------------------------------------
    """
    if cu_seqlens is not None:
        raise NotImplementedError("variable-length (cu_seqlens) forward is out of scope")

    B, T, H, K = r.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    orig_dtype = r.dtype
    rf = r.float() * scale
    kf = k.float()
    vf = v.float()
    wf = w.float()
    uf = u.float()

    state = rf.new_zeros(B, H, K, V)
    if initial_state is not None:
        state = state + initial_state.float()

    o = rf.new_zeros(B, T, H, V)
    for t in range(T):
        r_t = rf[:, t]                       # [B, H, K]
        k_t = kf[:, t]                       # [B, H, K]
        v_t = vf[:, t]                       # [B, H, V]
        w_t = wf[:, t].exp()                 # [B, H, K]
        kv = k_t[..., None] * v_t[:, :, None, :]          # [B, H, K, V]
        read = state + uf[None, :, :, None] * kv          # current-token bonus on kv
        o[:, t] = (read * r_t[..., None]).sum(-2)         # [B, H, V]
        state = state * w_t[..., None] + kv               # data-dependent decay + rank-1 add

    ht = state if output_final_state else None
    return o.to(orig_dtype), ht
