# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F

__all__ = ['chunk_ttt_linear']


def _ttt_linear_scan(q, k, v, w, b, eta, scale, eps, mini_batch_size,
                     initial_state, initial_state_bias, output_final_state):
    """The repo's own naive per-mini-batch TTT-linear recurrence (``[B, H, T, D]``
    layout). One Python iteration per mini-batch of ``mini_batch_size`` tokens; each
    step reconstructs the target through an inner-loop test-time gradient step with a
    layer-norm objective. Correct but slow: NT = T / mini_batch_size sequential
    iterations, each materialising dense per-batch matmuls."""
    B, H, T, D = q.shape
    BT = mini_batch_size
    NT = T // BT
    _q = (q * scale).reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _k = k.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _v = v.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _eta = eta.reshape(B, H, NT, BT, 1).permute(2, 0, 1, 3, 4)
    w = w.reshape(H, 1, D).to(torch.float32)
    b = b.reshape(H, 1, D).to(torch.float32)

    h = torch.zeros((B, H, D, D), device=v.device, dtype=torch.float32) if initial_state is None else initial_state
    hb = torch.zeros((B, H, 1, D), device=v.device, dtype=torch.float32) if initial_state_bias is None else initial_state_bias
    o = torch.empty_like(_v)

    for i in range(NT):
        q_i, k_i, v_i, eta_i = [x[i] for x in [_q, _k, _v, _eta]]
        kh = k_i @ h + hb
        reconstruction_target = v_i - k_i

        mean = kh.mean(-1, True)
        var = kh.var(-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        kh_hat = (kh - mean) / rstd

        g = w * kh_hat + b - reconstruction_target
        g = g * w
        v_new = (D * g - g.sum(-1, True) - kh_hat * (g * kh_hat).sum(-1, True)) / (rstd * D)

        Attn = torch.tril(q_i @ k_i.transpose(-2, -1))
        o_i = q_i @ h - (eta_i * Attn) @ v_new + hb - torch.tril(eta_i.expand_as(Attn)) @ v_new
        h = h - (eta_i[:, :, -1, :, None] * k_i).transpose(-1, -2) @ v_new
        hb = hb - torch.sum(eta_i[:, :, -1, :, None] * v_new, dim=-2, keepdim=True)

        mean = o_i.mean(dim=-1, keepdim=True)
        var = o_i.var(dim=-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        o[i] = o_i + (o_i - mean) / rstd * w + b

    o = o.permute(1, 2, 0, 3, 4).reshape(B, H, T, D)
    h = h if output_final_state else None
    hb = hb if output_final_state else None
    return o, h, hb


@torch.compiler.disable
def chunk_ttt_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: torch.Tensor | None = None,
    initial_state_bias: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
):
    r"""
    Test-Time-Training (TTT) linear layer forward.

    Args:
        q, k (torch.Tensor): queries / keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]` (`V == K`).
        w, b (torch.Tensor): output layer-norm weight / bias of shape `[H, V]`.
        eta (torch.Tensor): inner-loop learning rate of shape `[B, T, H, 1]` (or a float).
        scale (Optional[float]): query scale; defaults to `1 / sqrt(K)`.
        eps (float): layer-norm epsilon. Default: 1e-6.
        chunk_size (int): mini-batch size for the TTT inner loop. Default: 16.
        initial_state (Optional[torch.Tensor]): initial state `[N, H, K, V]`.
        initial_state_bias (Optional[torch.Tensor]): initial state bias `[N, H, 1, V]`.
        output_final_state (bool): whether to also return the final state (+ bias).

    Returns:
        o (torch.Tensor): outputs of shape `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]): `[N, H, K, V]` if requested else None.
        final_state_bias (Optional[torch.Tensor]): `[N, H, 1, V]` if requested else None.

    ---------------------------------------------------------------------------
    SLOW-BUT-CORRECT forward: an idiomatic eager TTT-linear scan (the repo's own
    ``chunk_ttt_linear_ref`` semantics) that walks the sequence one mini-batch at a
    time, materialising the dense per-mini-batch reconstruction, inner-loop
    gradient step, and output layer norm in plain torch. Numerically correct and
    autograd-differentiable, but run time is dominated by the number of dependent
    mini-batch steps (NT = T / chunk_size sequential iterations). Preserves the
    public (o, final_state, final_state_bias) contract. Making this forward's run
    time stop growing linearly with the number of dependent steps is the task.
    ---------------------------------------------------------------------------
    """
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if cu_seqlens is not None:
        raise NotImplementedError("variable-length (cu_seqlens) forward is out of scope")
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."

    # public [B, T, H, D] -> internal [B, H, T, D]
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    etah = eta.transpose(1, 2)
    T = qh.shape[-2]
    padded = (chunk_size - (T % chunk_size)) % chunk_size
    if padded > 0:
        qh = F.pad(qh, (0, 0, 0, padded))
        kh = F.pad(kh, (0, 0, 0, padded))
        vh = F.pad(vh, (0, 0, 0, padded))
        etah = F.pad(etah, (0, 0, 0, padded))
        etah[:, :, -1, :] = etah[:, :, -(padded + 1), :]
    qh, kh, vh, etah, wf, bf = map(lambda x: x.to(torch.float32), [qh, kh, vh, etah, w, b])
    o, final_state, final_state_bias = _ttt_linear_scan(
        qh, kh, vh, wf, bf, etah, scale, eps, chunk_size,
        initial_state, initial_state_bias, output_final_state,
    )
    o = o[:, :, :T, :].contiguous().transpose(1, 2)
    return o.to(q.dtype), final_state, final_state_bias
