"""Apply a frozen linear layer plus a scaled low-rank correction.

Public entry point:
    ``lowrank_adapter_apply(x, base_weight, factor_a, factor_b, scale) -> y``

A frozen base linear layer (weight ``base_weight`` of shape ``[N, K]``) is combined
with a small trainable low-rank correction built from two factors ``factor_a``
``[r, K]`` and ``factor_b`` ``[N, r]`` (with the shared inner dimension ``r`` much
smaller than ``N`` and ``K``) and a scalar ``scale``. For an input ``x`` whose last
dimension is ``K`` the result is::

    y = x @ base_weight.T + scale * ( (x @ factor_a.T) @ factor_b.T )

i.e. the frozen linear output plus a rank-``r`` correction.

Contract
--------
- ``x``: shape ``(..., K)`` (at least 2-D; the leading dims are token/batch axes),
  dtype ``torch.bfloat16`` or ``torch.float32``, CUDA tensor. ``K`` is the last
  (feature) dimension.
- ``base_weight``: shape ``(N, K)``, same floating dtype and device as ``x`` — the
  frozen linear weight, laid out row-major ``[out, in]`` like a standard linear.
- ``factor_a``: shape ``(r, K)``, same dtype and device as ``x``.
- ``factor_b``: shape ``(N, r)``, same dtype and device as ``x``. The shared inner
  dimension is ``r == factor_a.shape[0] == factor_b.shape[1]``.
- ``scale``: python ``float`` multiplying the low-rank correction.

Functionality::

    y = x @ base_weight.T + scale * ( (x @ factor_a.T) @ factor_b.T )

The base weight is frozen; only the two small factors carry the correction. The
result is linear in ``x`` (there is no bias term).

Returns
-------
``y``: shape ``(..., N)``, dtype matching ``x``.

Error contract
--------------
- ``TypeError`` if any of ``x`` / ``base_weight`` / ``factor_a`` / ``factor_b`` is
  not a floating (bfloat16/float32) ``torch.Tensor``, or if their dtypes differ
  from ``x``.
- ``ValueError`` for shape violations: ``x`` not at least 2-D; ``base_weight`` not
  ``(N, K)`` with ``K`` matching ``x``'s last dim; ``factor_a`` not ``(r, K)``;
  ``factor_b`` not ``(N, r)`` — in particular ``factor_a``'s row count must equal
  ``factor_b``'s column count (the shared rank ``r``).

Note on the current implementation
-----------------------------------
The current implementation first builds the full ``[N, K]`` correction matrix
``delta = scale * (factor_b @ factor_a)``, adds it into the base weight, and then
runs one ``[.., K] @ [K, N]`` matmul against the combined weight. Forming, storing
and re-reading that full ``[N, K]`` matrix is precisely the work the low-rank
structure lets you avoid: for ``r`` much smaller than ``N`` and ``K`` the correction
can be applied as two small matmuls, without ever materializing the ``[N, K]``
intermediate. It is correct but leaves the low-rank compute/memory saving on the
table.
"""

import torch


def _validate(x, base_weight, factor_a, factor_b, scale):
    if not isinstance(x, torch.Tensor) or not isinstance(base_weight, torch.Tensor) \
            or not isinstance(factor_a, torch.Tensor) or not isinstance(factor_b, torch.Tensor):
        raise TypeError("x, base_weight, factor_a, factor_b must be torch.Tensor")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"x must be bfloat16 or float32, got {x.dtype}")
    for name, t in (("base_weight", base_weight), ("factor_a", factor_a), ("factor_b", factor_b)):
        if t.dtype != x.dtype:
            raise TypeError(f"{name} dtype {t.dtype} must match x dtype {x.dtype}")
    if x.dim() < 2:
        raise ValueError(f"x must be at least 2-D (..., K), got {x.dim()}-D")
    K = x.shape[-1]
    if base_weight.dim() != 2 or base_weight.shape[1] != K:
        raise ValueError(f"base_weight must be 2-D (N, K) with K={K}, got shape {tuple(base_weight.shape)}")
    N = base_weight.shape[0]
    if factor_a.dim() != 2 or factor_a.shape[1] != K:
        raise ValueError(f"factor_a must be 2-D (r, K) with K={K}, got shape {tuple(factor_a.shape)}")
    r = factor_a.shape[0]
    if factor_b.dim() != 2 or factor_b.shape[0] != N or factor_b.shape[1] != r:
        raise ValueError(
            f"factor_b must be 2-D (N, r) with N={N} and r={r} (factor_a's row count), "
            f"got shape {tuple(factor_b.shape)}")
    return N, K, r


def lowrank_adapter_apply(x, base_weight, factor_a, factor_b, scale):
    """See module docstring for the full contract.

    Naive materialized-delta reference: build the full [N, K] correction
    ``scale * (factor_b @ factor_a)``, fold it into the base weight, then run a
    single matmul against the combined weight. Correct, but forms and stores the
    full [N, K] matrix the low-rank structure would let you skip.
    """
    _validate(x, base_weight, factor_a, factor_b, scale)
    delta = float(scale) * (factor_b @ factor_a)          # [N, r] @ [r, K] -> [N, K]
    merged = base_weight + delta                          # [N, K]
    y = x @ merged.transpose(-1, -2)                      # [.., K] @ [K, N] -> [.., N]
    return y
