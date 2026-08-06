"""Per-channel trailing-window weighted combination with left zero-padding.

Public entry point:
    ``channel_window_op(x, w, bias) -> y``

For each independent row (channel ``c``) of a ``(B, C, L)`` input, a short
length-``K`` weight vector ``w[c]`` is combined with a *trailing* window along the
length axis: output position ``t`` is the weighted sum of input positions
``t-K+1 .. t``. Positions with a negative index are treated as zero, i.e. the
sequence is left-padded by ``K-1`` so that the output length equals the input
length ``L`` and position ``t`` never reads any input beyond ``t``. A per-row bias
is then added and a smooth gating activation ``a * sigmoid(a)`` is applied
element-wise. Rows never interact — output row ``c`` depends only on ``x[:, c, :]``
and ``w[c, :]`` (and ``bias[c]``).

Contract
--------
- ``x``: shape ``(B, C, L)`` (batch, rows, length), dtype ``torch.float32``,
  ``torch.bfloat16`` or ``torch.float16``, CUDA tensor.
- ``w``: shape ``(C, K)`` — the per-row length-``K`` weights, same floating dtype as
  ``x``. ``K`` is small (typically 2-4) and ``K >= 1``.
- ``bias``: shape ``(C,)`` or ``None`` — per-row bias, same dtype as ``x``.

Functionality, for every ``(b, c, t)`` with ``xpad`` = ``x`` left-padded by ``K-1``
zeros along the length axis::

    a[b,c,t] = (bias[c] if bias is not None else 0)
               + sum_{j=0}^{K-1} w[c, j] * xpad[b, c, t + j]      # xpad[..,t+j] == x[.., t-(K-1)+j]
    y[b,c,t] = a[b,c,t] * sigmoid(a[b,c,t])                        # SiLU gating

The weighted sum is accumulated in float32 for stability and the result is cast
back to the dtype of ``x``. Output shape/dtype match ``x``.

Returns
-------
``y``: shape ``(B, C, L)``, dtype of ``x``.

Error contract
--------------
- ``TypeError`` if ``x`` or ``w`` is not a floating (float32/bfloat16/float16)
  ``torch.Tensor``, if ``w``/``bias`` dtype differs from ``x``, or if ``bias`` is
  neither ``None`` nor a ``torch.Tensor``.
- ``ValueError`` if ``x`` is not 3-D, if ``w`` is not 2-D with first dim ``C``
  (== ``x.shape[1]``), if the window length ``K`` (``w.shape[1]``) is ``< 1``, or if
  ``bias`` is not 1-D of length ``C``.

Note on allowed operations
--------------------------
The framework's built-in 1-D convolution primitives are out of scope for this task
— the operation is built explicitly. The current implementation first materializes
a left-padded copy of the input, then accumulates the ``K`` taps one at a time
(each tap a separate full-tensor read/modify/write that round-trips the whole
``(B, C, L)`` intermediate through global memory), and finally runs the activation
as one more pass. It is correct but leaves a lot of GPU memory bandwidth and
kernel-launch overhead on the table.
"""

import torch


def _validate(x, w, bias):
    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor):
        raise TypeError("x and w must be torch.Tensor")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"x must be float32/bfloat16/float16, got {x.dtype}")
    if w.dtype != x.dtype:
        raise TypeError(f"w dtype {w.dtype} must match x dtype {x.dtype}")
    if x.dim() != 3:
        raise ValueError(f"x must be 3-D (B, C, L), got {x.dim()}-D")
    B, C, L = x.shape
    if w.dim() != 2 or w.shape[0] != C:
        raise ValueError(f"w must be 2-D (C, K) with C={C}, got shape {tuple(w.shape)}")
    K = int(w.shape[1])
    if K < 1:
        raise ValueError(f"window length K must be >= 1, got {K}")
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a torch.Tensor or None")
        if bias.dtype != x.dtype:
            raise TypeError(f"bias dtype {bias.dtype} must match x dtype {x.dtype}")
        if bias.dim() != 1 or bias.shape[0] != C:
            raise ValueError(f"bias must be 1-D of length C={C}, got shape {tuple(bias.shape)}")
    return B, C, L, K


def channel_window_op(x, w, bias):
    """See module docstring for the full contract.

    Naive multi-pass reference: materialize a left-padded copy, accumulate one tap
    per iteration (each a full-tensor pass through global memory), add the bias,
    then apply the gating activation in a final separate pass. Correct but
    memory/launch-bound.
    """
    B, C, L, K = _validate(x, w, bias)
    xf = x.to(torch.float32)
    # explicit left zero-pad by K-1 (materializes a padded copy in memory)
    xp = torch.nn.functional.pad(xf, (K - 1, 0))
    # accumulate one tap at a time; each step reads/writes the full (B, C, L) tensor
    acc = torch.zeros((B, C, L), dtype=torch.float32, device=x.device)
    for j in range(K):
        wj = w[:, j].to(torch.float32).view(1, C, 1)
        acc = acc + wj * xp[:, :, j:j + L]
    if bias is not None:
        acc = acc + bias.to(torch.float32).view(1, C, 1)
    # separate activation pass over the full tensor
    y = acc * torch.sigmoid(acc)
    return y.to(x.dtype)
