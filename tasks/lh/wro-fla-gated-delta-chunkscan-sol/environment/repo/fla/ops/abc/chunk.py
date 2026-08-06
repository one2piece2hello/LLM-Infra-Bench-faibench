# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ABC (Attention with Bounded-memory Control): a two-stage gated linear
# recurrence with a memory of M slots. Stage 1 reads queries against a
# key/slot memory (hk) to produce slot logits; a softmax over the M slots
# turns them into slot probabilities; stage 2 reads those probabilities
# against a slot/value memory (hv) to produce the output. The gate is derived
# from the slot logits via a cumulative log-sum-exp normalizer.
#
# Public entry: chunk_abc(q, k, v, s, initial_state, output_final_state,
# head_first) -> (o, final_state).

import torch

from fla.utils import input_guard


class ChunkABCFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, s, initial_state, output_final_state):
        # -----------------------------------------------------------------
        # SLOW-BUT-CORRECT forward: an idiomatic eager per-timestep evaluation
        # of the two-stage ABC recurrence, written in plain torch. It is
        # numerically correct but evaluates the slot-memory and value-memory
        # scans strictly one time step at a time over the T positions -> O(T)
        # sequential work with no time-parallelism. Preserves the public
        # (o, final_state) contract of chunk_abc. Making this forward fast is
        # the task. Inputs are head-first [B, H, T, *].
        # -----------------------------------------------------------------
        B, H, T, K = q.shape
        V = v.shape[-1]
        M = s.shape[-1]
        scale = K ** -0.5

        qf = q.float()
        kf = k.float()
        vf = v.float()
        sf = s.float()

        # gate from slot logits via cumulative log-sum-exp normalizer:
        #   z = logcumsumexp(s, dim=T); g = shift(z) - z; s_p = exp(s - z)
        z = sf.logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        sp = torch.exp(sf - z)

        hk = qf.new_zeros(B, H, K, M)
        if initial_state is not None:
            hk = hk + initial_state[0].float()
        ok = torch.zeros_like(sf)
        for i in range(T):
            q_i = qf[:, :, i] * scale
            k_i = kf[:, :, i]
            v_i = sp[:, :, i]
            g_i = g[:, :, i].exp()
            hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
            ok[:, :, i] = (q_i[..., None] * hk).sum(-2)

        qv = ok.softmax(-1)

        hv = qf.new_zeros(B, H, M, V)
        if initial_state is not None:
            hv = hv + initial_state[1].float()
        ov = torch.zeros_like(vf)
        for i in range(T):
            q_i = qv[:, :, i]
            k_i = sp[:, :, i]
            v_i = vf[:, :, i]
            g_i = g[:, :, i].exp()
            hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
            ov[:, :, i] = (q_i[..., None] * hv).sum(-2)

        final_state = None
        if output_final_state:
            final_state = (hk, hv)
        ov = ov.to(q.dtype)
        ctx.save_for_backward(q, k, v, s)
        return ov, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dov, dht=None):
        # The evaluation benchmark exercises the forward path only. A gradient
        # is not part of the measured contract for this task.
        raise NotImplementedError("backward is out of scope for this task")


@torch.compiler.disable
def chunk_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    initial_state: tuple[torch.Tensor] | None = None,
    output_final_state: bool = False,
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not head_first:
        q, k, v, s = map(lambda x: x.transpose(1, 2), (q, k, v, s))
    o, final_state = ChunkABCFunction.apply(q, k, v, s, initial_state, output_final_state)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
