"""Causal sequence mixer with a gated (decaying) running key-value state.

Public entry point:
    ``gated_state_recurrence(q, k, v, g, initial_state=None,
                             output_final_state=False)``

This layer mixes a sequence of tokens through a small running state matrix rather
than through pairwise attention. For each ``(batch, head)`` it maintains a state
matrix ``S`` of shape ``(Dk, Dv)`` that is updated one position at a time, left to
right (causal). What makes it expressive is a per-position, per-key-feature
multiplicative **gate**: before each new key/value pair is written, every row of
the state is scaled down by its own decay factor, so the state remembers recent
associations more strongly than old ones (a learned, per-feature forget rate).

Contract
--------
- ``q``, ``k``: shape ``(B, H, L, Dk)`` — per-position query and key. Same
  floating dtype (``torch.bfloat16`` or ``torch.float16``), CUDA tensors.
- ``v``: shape ``(B, H, L, Dv)`` — per-position value, same dtype/device.
- ``g``: shape ``(B, H, L, Dk)`` — per-position, per-key-feature gate in
  **log space**; the multiplicative decay applied to the state is ``exp(g)``
  (so ``g <= 0`` decays, ``g == 0`` keeps the state, very negative ``g`` nearly
  forgets it). Same dtype/device as ``q``.
- ``initial_state``: optional ``(B, H, Dk, Dv)`` starting state (same device;
  read in float32). ``None`` means a zero state.
- ``output_final_state``: if ``True`` also return the final state ``S``.

Semantics (per ``(batch, head)``, state ``S`` of shape ``(Dk, Dv)``)::

    scale = Dk ** -0.5
    S = initial_state (or zeros)                       # float32
    for t = 0 .. L-1:
        a_t = exp(g_t)                                 # (Dk,) per-feature decay
        S   = diag(a_t) @ S + outer(k_t, v_t)          # decay the state, then write
        o_t = (scale * q_t)^T S                        # query reads the updated state

where ``diag(a_t) @ S`` scales row ``d`` of the state by ``a_t[d]``
(``S[d, m] <- a_t[d] * S[d, m]``), ``outer(k_t, v_t)[d, m] = k_t[d] * v_t[m]`` and
the query readout contracts ``Dk`` (``o_t[m] = sum_d scale * q_t[d] * S[d, m]``).
The gate scales the **carried state**, not the freshly written pair. The state and
the readout are accumulated in float32 for stability; the output is cast back to
the input dtype. The recurrence is strictly causal: ``o_t`` depends only on
positions ``<= t``.

Returns
-------
- ``o``: shape ``(B, H, L, Dv)``, dtype of ``q``.
- ``(o, final_state)`` when ``output_final_state`` is ``True`` — ``final_state``
  is ``(B, H, Dk, Dv)`` float32.

Error contract
--------------
- ``TypeError`` if any of ``q, k, v, g`` is not a floating (bfloat16/float16)
  ``torch.Tensor``, or their dtypes differ from ``q``.
- ``ValueError`` if ``q`` is not 4-D, ``k`` shape != ``q`` shape, ``g`` shape !=
  ``q`` shape, ``v`` is not ``(B, H, L, Dv)`` sharing ``(B, H, L)`` with ``q``, or
  ``initial_state`` (when given) is not ``(B, H, Dk, Dv)``.

Note on allowed operations
--------------------------
Third-party sequence-mixing / attention / state-recurrence library operators are
out of scope — the state recurrence is built here from primitive tensor
operations. The current implementation walks the sequence one position at a time
in Python: every position launches its own small GPU operations and each state
update must wait for the previous one, so the sequential dependency chain
dominates the runtime even though the per-position work is tiny.
"""

import torch


def _validate(q, k, v, g, initial_state):
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"q must be bfloat16 or float16, got {q.dtype}")
    for name, t in (("k", k), ("v", v), ("g", g)):
        if t.dtype != q.dtype:
            raise TypeError(f"{name} dtype {t.dtype} must match q dtype {q.dtype}")
    if q.dim() != 4:
        raise ValueError(f"q must be 4-D (B, H, L, Dk), got {q.dim()}-D")
    if tuple(k.shape) != tuple(q.shape):
        raise ValueError(f"k shape {tuple(k.shape)} must equal q shape {tuple(q.shape)}")
    if tuple(g.shape) != tuple(q.shape):
        raise ValueError(f"g shape {tuple(g.shape)} must equal q shape {tuple(q.shape)}")
    B, H, L, Dk = q.shape
    if v.dim() != 4 or tuple(v.shape[:3]) != (B, H, L):
        raise ValueError(f"v must be (B, H, L, Dv) sharing (B, H, L)={(B, H, L)}, got {tuple(v.shape)}")
    Dv = v.shape[-1]
    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor or None")
        if tuple(initial_state.shape) != (B, H, Dk, Dv):
            raise ValueError(
                f"initial_state must be (B, H, Dk, Dv)={(B, H, Dk, Dv)}, got {tuple(initial_state.shape)}")
    return B, H, L, Dk, Dv


def gated_state_recurrence(q, k, v, g, initial_state=None, output_final_state=False):
    """See module docstring for the full contract.

    Naive sequential reference: a Python loop over the L positions, each decaying
    the state by its per-feature gate, writing the current key/value outer product,
    and reading it back for the query. Correct but the per-position dependency
    chain makes it slow (many tiny launches, no reuse of the large matmul units).
    """
    B, H, L, Dk, Dv = _validate(q, k, v, g, initial_state)
    orig_dtype = q.dtype
    qf = q.to(torch.float32) * (Dk ** -0.5)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    af = g.to(torch.float32).exp()                         # per-feature decay factor

    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(torch.float32)
    o = torch.empty(B, H, L, Dv, dtype=torch.float32, device=q.device)

    for t in range(L):
        q_t = qf[:, :, t]                                  # (B, H, Dk)
        k_t = kf[:, :, t]                                  # (B, H, Dk)
        v_t = vf[:, :, t]                                  # (B, H, Dv)
        a_t = af[:, :, t].unsqueeze(-1)                    # (B, H, Dk, 1) decay
        S = a_t * S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # decay, then write outer(k, v)
        o[:, :, t] = (q_t.unsqueeze(-1) * S).sum(dim=-2)   # query readout of updated state

    o = o.to(orig_dtype)
    if output_final_state:
        return o, S
    return o
