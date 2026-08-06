"""Scaled-dot-product causal attention over a full sequence.

Public entry point:
    ``causal_attention(q, k, v, scale, causal=True) -> out``

This is the attention step used inside a transformer block while processing a
whole input sequence at once. For every (batch, query-head, query-position) it
forms a probability distribution over key-positions from the scaled
query-key similarities, then returns the probability-weighted average of the
value rows.

Contract
--------
- ``q``: shape ``(B, H, S, D)``, dtype ``torch.bfloat16`` or ``torch.float16``,
  CUDA tensor. ``H`` is the number of query heads, ``S`` the sequence length,
  ``D`` the per-head feature width.
- ``k``, ``v``: shape ``(B, Hk, S, D)``, same dtype and device as ``q``. ``Hk``
  is the number of key/value heads. ``H`` must be an integer multiple of ``Hk``
  (each group of ``H // Hk`` query heads shares one key/value head); ``Hk == H``
  means every head is independent.
- ``scale``: python ``float`` applied to the query-key similarities before the
  softmax (typically ``1/sqrt(D)``).
- ``causal``: ``bool``. When ``True`` a query at position ``i`` may attend only
  to key positions ``j <= i``; when ``False`` it attends to all positions.

Functionality, for each (batch ``b``, query-head ``h``, query-position ``i``),
with key/value head ``hk = h // (H // Hk)``::

    logits[j] = scale * dot(q[b, h, i, :], k[b, hk, j, :])   # over feature dim
    logits[j] = -inf   if causal and j > i
    p[j]      = softmax(logits)[j]                            # over key axis
    out[b, h, i, :] = sum_j p[j] * v[b, hk, j, :]

Similarities and the softmax normalization are accumulated in float32 for
numerical stability; the result is cast back to the input dtype. Each query row
is normalized independently and row/position order is preserved.

Returns
-------
``out``: shape ``(B, H, S, D)``, dtype matching ``q``.

Error contract
--------------
- ``TypeError`` if ``q``, ``k`` or ``v`` is not a floating (bfloat16/float16)
  ``torch.Tensor``, or ``k``/``v`` dtype differs from ``q``.
- ``ValueError`` if ``q``/``k``/``v`` is not 4-D, if the ``B``/``S``/``D`` axes of
  ``k``/``v`` do not match ``q``, if ``k`` and ``v`` shapes differ, or if ``H`` is
  not an integer multiple of ``Hk``.

Note on allowed operations and memory
-------------------------------------
The framework's built-in fused attention primitives are out of scope for this
task -- the attention is built explicitly. The current implementation forms the
full ``(S, S)`` similarity matrix for every (batch, head), applies the mask,
normalizes it, and multiplies by the values. It is correct but its peak extra
memory grows with ``B * H * S * S`` -- quadratic in the sequence length -- so it
leaves a lot of GPU memory on the table at long sequences.
"""

import torch


def _validate(q, k, v):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"q must be bfloat16 or float16, got {q.dtype}")
    if k.dtype != q.dtype:
        raise TypeError(f"k dtype {k.dtype} must match q dtype {q.dtype}")
    if v.dtype != q.dtype:
        raise TypeError(f"v dtype {v.dtype} must match q dtype {q.dtype}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must all be 4-D (B, H, S, D) / (B, Hk, S, D)")
    B, H, S, D = q.shape
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError(f"k shape {tuple(k.shape)} must equal v shape {tuple(v.shape)}")
    Bk, Hk, Sk, Dk = k.shape
    if Bk != B or Sk != S or Dk != D:
        raise ValueError(
            f"k/v must share B,S,D with q: got k {tuple(k.shape)} vs q {tuple(q.shape)}")
    if Hk <= 0 or H % Hk != 0:
        raise ValueError(f"H={H} must be a positive integer multiple of Hk={Hk}")
    return B, H, S, D, Hk


def causal_attention(q, k, v, scale, causal=True):
    """See module docstring for the full contract.

    Naive dense reference: materialize the full ``(S, S)`` similarity matrix per
    (batch, head), mask, softmax, and multiply by the values. Correct but its
    peak memory is quadratic in the sequence length.
    """
    B, H, S, D, Hk = _validate(q, k, v)
    group = H // Hk

    qf = q.to(torch.float32)
    # broadcast each key/value head to its group of query heads
    kf = k.to(torch.float32).repeat_interleave(group, dim=1)   # (B, H, S, D)
    vf = v.to(torch.float32).repeat_interleave(group, dim=1)   # (B, H, S, D)

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * float(scale)   # (B, H, S, S)
    if causal:
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)                            # (B, H, S, S)
    out = torch.matmul(probs, vf)                                    # (B, H, S, D)
    return out.to(q.dtype)
