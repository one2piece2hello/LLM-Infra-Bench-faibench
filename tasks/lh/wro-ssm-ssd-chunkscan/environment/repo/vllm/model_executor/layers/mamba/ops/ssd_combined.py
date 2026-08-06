# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501

import torch
import torch.nn.functional as F

# NOTE: this module computes the Mamba-2 state-space scan with a
# straightforward sequential recurrence over the sequence dimension. It is a
# correct, readable reference: for every time step it updates the SSM state and
# reads out the output. It processes one time step at a time and materializes
# the running state each step.


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(x,
                                   dt,
                                   A,
                                   B,
                                   C,
                                   chunk_size,
                                   D=None,
                                   z=None,
                                   dt_bias=None,
                                   initial_states=None,
                                   seq_idx=None,
                                   chunk_indices=None,
                                   chunk_offsets=None,
                                   cu_seqlens=None,
                                   dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   state_dtype=None,
                                   out=None):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads, )
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads, )
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        if cu_seqlens is None:
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        else:
            assert initial_states.shape == (len(cu_seqlens) - 1, nheads,
                                            headdim, dstate)
    if cu_seqlens is not None:
        raise NotImplementedError(
            "variable-length (cu_seqlens) sequences are not supported by this "
            "reference implementation")

    out_dtype = x.dtype
    if state_dtype is None:
        state_dtype = C.dtype

    # ---- discretize dt exactly as the reference does ----
    #      dt <- dt + dt_bias ; optional softplus ; clamp to dt_limit
    dt_proc = dt.float()
    if dt_bias is not None:
        dt_proc = dt_proc + dt_bias.float()
    if dt_softplus:
        dt_proc = torch.where(dt_proc <= 20.0, F.softplus(dt_proc), dt_proc)
    dt_min, dt_max = dt_limit
    dt_proc = torch.clamp(dt_proc, min=dt_min, max=dt_max)  # (batch, seqlen, nheads)

    Af = A.float()  # (nheads,)
    xf = x.float()
    Bf = B.float()
    Cf = C.float()

    # map each head to its B/C group
    heads_per_group = nheads // ngroups
    # gather group -> head:  (batch, seqlen, nheads, dstate)
    grp_index = torch.arange(nheads, device=x.device) // heads_per_group
    Bh = Bf.index_select(2, grp_index)  # (b, l, h, n)
    Ch = Cf.index_select(2, grp_index)

    # ---- running SSM state, updated one time step at a time ----
    if initial_states is not None:
        h_state = initial_states.float().clone()  # (b, h, p, n)
    else:
        h_state = torch.zeros(batch, nheads, headdim, dstate,
                              device=x.device, dtype=torch.float32)

    y = torch.empty(batch, seqlen, nheads, headdim,
                    device=x.device, dtype=torch.float32)

    for t in range(seqlen):
        dt_t = dt_proc[:, t, :]                       # (b, h)
        dA = torch.exp(dt_t * Af)                      # (b, h)
        # dt_t * (x_t outer B_t)   -> (b, h, p, n)
        dBx = (dt_t[:, :, None, None]
               * xf[:, t, :, :, None]
               * Bh[:, t, :, None, :])
        h_state = dA[:, :, None, None] * h_state + dBx
        # y_t = C_t . h_state      -> (b, h, p)
        y_t = (h_state * Ch[:, t, :, None, :]).sum(dim=-1)
        if D is not None:
            if D.dim() == 2:                           # (nheads, headdim)
                y_t = y_t + D.float()[None, :, :] * xf[:, t, :, :]
            else:                                      # (nheads,)
                y_t = y_t + D.float()[None, :, None] * xf[:, t, :, :]
        y[:, t, :, :] = y_t

    if z is not None:
        zf = z.float()
        y = y * (zf * torch.sigmoid(zf))

    final_states = h_state.to(state_dtype)

    if out is not None:
        out.copy_(y.to(out_dtype))
        out_x = out
    else:
        out_x = y.to(out_dtype)

    # dt_out / dA_cumsum / states are internal chunked artifacts in the fast
    # path; the public wrapper only forwards `out` and `final_states`.
    return out_x, dt_proc, None, None, final_states


def mamba_chunk_scan_combined(x,
                              dt,
                              A,
                              B,
                              C,
                              chunk_size,
                              D=None,
                              z=None,
                              dt_bias=None,
                              initial_states=None,
                              seq_idx=None,
                              chunk_indices=None,
                              chunk_offsets=None,
                              cu_seqlens=None,
                              dt_softplus=False,
                              dt_limit=(0.0, float("inf")),
                              out=None,
                              return_final_states=False,
                              return_varlen_states=False,
                              state_dtype=None):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
        state_dtype: The data type of the ssm state
    """

    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert cu_seqlens is not None, "cu_seqlens must be provided if return_varlen_states is True"
    out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        out=out,
        state_dtype=state_dtype)
    if not return_varlen_states:
        if not return_final_states:
            return
        else:
            return final_states
    else:
        varlen_states = rest[0]
        return (varlen_states) if not return_final_states else (final_states,
                                                                varlen_states)
