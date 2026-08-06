"""First-order linear state-space scan (selective SSM recurrence).

Public entry point:
    ``state_space_scan(A, B, C, x) -> y``

This is the core sequence operator inside a state-space / linear-recurrence block.
A hidden state ``h`` of shape ``(B, D, N)`` is evolved along the time axis ``L`` by a
first-order linear recurrence whose per-timestep coefficients are *selective* (they
depend on the position ``t``): the state decays/mixes through ``A[t]`` and is driven
by the outer product of the input ``x[t]`` with the input projection ``B[t]``. The
per-timestep output ``y[t]`` reads the state out through the output projection
``C[t]``.

Contract
--------
- ``A``: shape ``(B, L, D, N)`` — the per-timestep state-transition coefficient
  (multiplicative, applied elementwise to the state). Floating dtype, CUDA tensor.
- ``B``: shape ``(B, L, N)`` — the per-timestep input projection.
- ``C``: shape ``(B, L, N)`` — the per-timestep output projection.
- ``x``: shape ``(B, L, D)`` — the input sequence.
- ``B`` (batch), ``L`` (time), ``D`` (channels), ``N`` (state width). All four
  tensors share dtype and device. Primary dtype ``torch.float32``; ``bfloat16`` /
  ``float16`` inputs are also accepted (the state is accumulated in float32).

Functionality, per batch element and per channel ``d`` / state index ``n``, marching
``t = 0, 1, ..., L-1`` along the time axis with initial state ``h[-1] = 0``::

    bx[t]         = x[t, :, None] * B[t, None, :]      # (D, N) input drive at step t
    h[t]          = A[t] * h[t-1] + bx[t]              # (D, N) first-order recurrence
    y[t, d]       = sum_n C[t, n] * h[t, d, n]         # (D,)  read-out through C[t]

The recurrence is **causal and strictly ordered in time**: ``h[t]`` (and therefore
``y[t]``) depends only on inputs at steps ``0..t``. The state is accumulated in
float32 for numerical stability; every ``(batch, d, n)`` track evolves independently
along ``t``.

Returns
-------
``y``: shape ``(B, L, D)``, dtype matching ``x``.

Error contract
--------------
- ``TypeError`` if any of ``A``, ``B``, ``C``, ``x`` is not a floating
  (float16/bfloat16/float32/float64) ``torch.Tensor``, or their dtypes differ.
- ``ValueError`` if the ranks are wrong (``x`` not 3-D, ``A`` not 4-D, ``B``/``C``
  not 3-D) or the shared dimensions ``(B, L, D, N)`` are inconsistent across inputs.

Note on allowed operations
--------------------------
Library sequence-scan primitives that would perform the whole recurrence in one call
are out of scope — the scan is built explicitly. The current implementation walks the
time axis in a plain Python ``for`` loop, one timestep per iteration, so it issues
O(L) dependent kernel launches and serializes on the time dimension. It is correct
but leaves the sequence-level parallelism of the recurrence on the table.
"""

import torch


def _validate(A, B, C, x):
    for name, t in (("A", A), ("B", B), ("C", C), ("x", x)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
    floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if x.dtype not in floating:
        raise TypeError(f"x must be a floating dtype, got {x.dtype}")
    for name, t in (("A", A), ("B", B), ("C", C)):
        if t.dtype != x.dtype:
            raise TypeError(f"{name} dtype {t.dtype} must match x dtype {x.dtype}")
    if x.dim() != 3:
        raise ValueError(f"x must be 3-D (B, L, D), got {x.dim()}-D")
    if A.dim() != 4:
        raise ValueError(f"A must be 4-D (B, L, D, N), got {A.dim()}-D")
    if B.dim() != 3 or C.dim() != 3:
        raise ValueError(f"B and C must be 3-D (B, L, N), got {B.dim()}-D and {C.dim()}-D")
    Bsz, L, D = x.shape
    if tuple(A.shape[:3]) != (Bsz, L, D):
        raise ValueError(f"A leading dims {tuple(A.shape[:3])} must equal (B, L, D)={(Bsz, L, D)}")
    N = A.shape[3]
    if tuple(B.shape) != (Bsz, L, N):
        raise ValueError(f"B shape {tuple(B.shape)} must be (B, L, N)={(Bsz, L, N)}")
    if tuple(C.shape) != (Bsz, L, N):
        raise ValueError(f"C shape {tuple(C.shape)} must be (B, L, N)={(Bsz, L, N)}")
    return Bsz, L, D, N


def state_space_scan(A, B, C, x):
    """See module docstring for the full contract.

    Naive sequential reference: march the time axis one step at a time in a Python
    loop, accumulating the (D, N) hidden state in float32. Correct but latency-bound
    on the sequential time dependency (O(L) dependent steps / kernel launches).
    """
    Bsz, L, D, N = _validate(A, B, C, x)
    out_dtype = x.dtype
    Af = A.to(torch.float32)
    Bf = B.to(torch.float32)
    Cf = C.to(torch.float32)
    xf = x.to(torch.float32)

    bx = xf.unsqueeze(-1) * Bf.unsqueeze(2)          # (B, L, D, N) input drive
    h = torch.zeros(Bsz, D, N, dtype=torch.float32, device=A.device)
    hs = []
    for t in range(L):                               # sequential O(L) time loop
        h = Af[:, t] * h + bx[:, t]                  # (B, D, N)
        hs.append(h)
    hs = torch.stack(hs, dim=1)                      # (B, L, D, N)
    y = (hs * Cf.unsqueeze(2)).sum(dim=-1)           # (B, L, D) read-out through C
    return y.to(out_dtype)
