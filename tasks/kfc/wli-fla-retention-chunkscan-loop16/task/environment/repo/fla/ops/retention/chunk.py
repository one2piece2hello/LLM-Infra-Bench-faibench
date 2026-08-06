# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Multi-scale retention (RetNet) chunked linear-attention entry point.
#
# Public entry `chunk_retention(q, k, v, scale, initial_state,
# output_final_state, cu_seqlens) -> (o, final_state)`.
#
# Retention is a linear-attention operator with a FIXED, data-independent
# per-head exponential decay
#       gamma[h] = 1 - 2 ** (-5 - h)
# and the state recurrence
#       S_t = gamma[h] * S_{t-1} + k_t^T v_t
#       o_t = scale * (q_t @ S_t)
# equivalently the causal decay-weighted attention
#       o_t = scale * sum_{s<=t} gamma[h]^(t-s) (q_t . k_s) v_s .

import torch


def _retention_recurrent_bthk(q, k, v, scale, gamma, initial_state, output_final_state):
    # -----------------------------------------------------------------
    # SLOW-BUT-CORRECT forward: an idiomatic eager per-timestep gated
    # linear-attention recurrence written in plain torch. It is numerically
    # correct and preserves the public (o, final_state) contract of
    # chunk_retention, but evaluates the scan strictly one time step at a time
    # over the T positions -> O(T) sequential kernel launches with no
    # time-parallelism (no chunked/blocked reformulation). Making this fast is
    # the task.
    # -----------------------------------------------------------------
    B, T, H, K = q.shape
    V = v.shape[-1]
    qf = q.float()
    kf = k.float()
    vf = v.float()
    g = gamma.view(1, H, 1, 1).float()               # [1,H,1,1] per-head decay
    S = qf.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = qf.new_zeros(B, T, H, V)
    for t in range(T):
        kt = kf[:, t]                                # [B,H,K]
        vt = vf[:, t]                                # [B,H,V]
        qt = qf[:, t]                                # [B,H,K]
        S = g * S + kt.unsqueeze(-1) * vt.unsqueeze(-2)   # [B,H,K,V]
        o[:, t] = scale * torch.einsum('bhk,bhkv->bhv', qt, S)
    final_state = S if output_final_state else None
    return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens=None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    kwargs.pop('chunk_size', None)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    H = q.shape[2]
    gamma = 1 - torch.pow(
        torch.tensor(2.0, dtype=torch.float, device=q.device),
        -5.0 - torch.arange(H, dtype=torch.float, device=q.device),
    )                                                # [H] fixed per-head decay

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
        # honest slow varlen path: process each sequence segment independently,
        # each with its own initial state and its own final state.
        bounds = cu_seqlens.tolist()
        outs = []
        finals = []
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            init_i = None if initial_state is None else initial_state[i:i + 1]
            o_i, fs_i = _retention_recurrent_bthk(
                q[:, s:e], k[:, s:e], v[:, s:e], scale, gamma, init_i, output_final_state,
            )
            outs.append(o_i)
            if output_final_state:
                finals.append(fs_i)
        o = torch.cat(outs, dim=1)
        final_state = torch.cat(finals, dim=0) if output_final_state else None
        return o, final_state

    return _retention_recurrent_bthk(q, k, v, scale, gamma, initial_state, output_final_state)
