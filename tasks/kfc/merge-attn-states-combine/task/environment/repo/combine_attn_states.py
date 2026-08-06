"""Combine partial attention outputs into the final attention result.

Public entry point:
    ``combine_attn_states(partial_out, partial_lse) -> (out, lse)``

A high-throughput inference server splits the attention over a long key/value
sequence into several independent chunks, attends within each chunk on its own, and
produces for every chunk a *partial* attention output together with the log-sum-exp
of that chunk's softmax scores (its log-domain normalizer). This routine recombines
those partial results into the single attention output that would have been obtained
by attending over the whole sequence at once.

Contract
--------
- ``partial_out``: shape ``(N, R, D)`` — ``N`` partial outputs, ``R`` independent
  rows (one per query token/head), ``D`` the head/feature dimension. dtype
  ``torch.float32``, ``torch.bfloat16`` or ``torch.float16``; CUDA tensor.
- ``partial_lse``: shape ``(N, R)`` — the matching log-sum-exp normalizer of each
  partial. Same dtype and device as ``partial_out``. A value of ``-inf`` marks a
  chunk that saw no keys (it contributes nothing to that row).
- No other arguments.

Functionality, for each row ``r`` independently, combining the ``N`` partials::

    m       = max_n partial_lse[n, r]                  # log-domain max (stability)
    w[n]    = exp(partial_lse[n, r] - m)               # per-chunk weight; -inf -> 0
    denom   = sum_n w[n]
    out[r]  = (sum_n w[n] * partial_out[n, r]) / denom # weighted average, float32
    lse[r]  = log(denom) + m                           # combined log-normalizer

If every chunk of a row is ``-inf`` (the row saw no keys at all) the combined output
for that row is all-zeros and its ``lse`` is ``-inf``.

Returns
-------
``(out, lse)``: ``out`` shape ``(R, D)`` dtype of ``partial_out``; ``lse`` shape
``(R,)`` dtype of ``partial_lse``. The weighted average is accumulated in float32 for
stability; each row is combined independently, row order is preserved, and the result
does not depend on the order of the ``N`` chunks.

Error contract
--------------
- ``TypeError`` if ``partial_out`` or ``partial_lse`` is not a floating
  (float32/bfloat16/float16) ``torch.Tensor``, or their dtypes differ.
- ``ValueError`` if ``partial_out`` is not 3-D ``(N, R, D)``, ``partial_lse`` is not
  2-D ``(N, R)``, or their leading ``(N, R)`` shapes disagree.

Note on allowed operations
--------------------------
The combine must be built explicitly from elementwise/reduction operations; a vendor
attention-state combine primitive is out of scope. The current implementation is
correct but memory-bound: it forms the full ``(N, R, D)`` array of weighted partials
as an intermediate in global memory and makes several separate passes over the data
(max, exponentiate, weight, reduce, divide), moving far more bytes than the result
itself requires.
"""

import torch


def _validate(partial_out, partial_lse):
    if not isinstance(partial_out, torch.Tensor) or not isinstance(partial_lse, torch.Tensor):
        raise TypeError("partial_out and partial_lse must be torch.Tensor")
    if partial_out.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(
            f"partial_out must be float32/bfloat16/float16, got {partial_out.dtype}")
    if partial_lse.dtype != partial_out.dtype:
        raise TypeError(
            f"partial_lse dtype {partial_lse.dtype} must match partial_out dtype {partial_out.dtype}")
    if partial_out.dim() != 3:
        raise ValueError(f"partial_out must be 3-D (N, R, D), got {partial_out.dim()}-D")
    if partial_lse.dim() != 2:
        raise ValueError(f"partial_lse must be 2-D (N, R), got {partial_lse.dim()}-D")
    if tuple(partial_lse.shape) != tuple(partial_out.shape[:2]):
        raise ValueError(
            f"partial_lse shape {tuple(partial_lse.shape)} must equal partial_out (N, R) "
            f"{tuple(partial_out.shape[:2])}")
    return partial_out.shape


def combine_attn_states(partial_out, partial_lse):
    """See module docstring for the full contract.

    Naive multi-pass reference: build the (N, R, D) weighted-partial tensor and reduce
    it. Correct but memory-bound (materializes an N*R*D intermediate and round-trips
    it through global memory across several separate kernel launches).
    """
    _validate(partial_out, partial_lse)
    out_f = partial_out.to(torch.float32)                     # (N, R, D)
    lse_f = partial_lse.to(torch.float32)                     # (N, R)

    m = lse_f.max(dim=0, keepdim=True).values                 # (1, R)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.exp(lse_f - m_safe)                             # (N, R); -inf chunk -> 0
    denom = w.sum(dim=0)                                      # (R,)

    weighted = w.unsqueeze(-1) * out_f                        # (N, R, D) big intermediate
    acc = weighted.sum(dim=0)                                 # (R, D)

    finite = denom > 0                                        # rows that saw at least one key
    safe_denom = torch.where(finite, denom, torch.ones_like(denom))
    out = torch.where(finite.unsqueeze(-1), acc / safe_denom.unsqueeze(-1),
                      torch.zeros_like(acc))
    lse = torch.where(finite, torch.log(safe_denom) + m_safe.squeeze(0),
                      torch.full_like(denom, float("-inf")))
    return out.to(partial_out.dtype), lse.to(partial_lse.dtype)
