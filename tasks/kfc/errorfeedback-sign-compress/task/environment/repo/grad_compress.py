"""Gradient compression with a carry-over residual for low-bandwidth data-parallel
training.

Public entry points:
    ``compress(buf, residual) -> (payload, new_residual)``
    ``decompress(payload) -> torch.Tensor``

During distributed training each worker must move its gradient buffer to the
other workers every step. Moving the full-precision buffer is bandwidth-hungry;
the goal of this module is to move a *compact* representation of the buffer and a
persistent ``residual`` accumulator so that, although each individual transmission
is lossy, the running average of what is transmitted stays unbiased over many
steps.

Contract
--------
- ``buf``: a gradient tensor of any shape (flattened internally), dtype
  ``torch.float32``, CUDA tensor.
- ``residual``: a persistent accumulator, dtype ``torch.float32``, with the
  **same number of elements** as ``buf`` (same shape). It carries the part of the
  buffer that the previous compression could not represent.
- ``compress`` returns ``(payload, new_residual)``:
    * ``payload``: a picklable representation of the compensated buffer, from
      which ``decompress`` reconstructs a full-shape tensor. The number of bytes
      the payload occupies is the value that is scored.
    * ``new_residual``: the updated accumulator, an fp32 tensor with the same
      shape as ``buf``.
- ``decompress(payload)`` reconstructs a full-shape (``buf.shape``) fp32 tensor.

Required semantics (for each call), writing ``comp = buf + residual``:

    let ``q = decompress(compress(buf, residual).payload)``
    then ``new_residual == comp - q``          # carry over what q could not represent

This accumulator identity must hold exactly (up to fp32 rounding). It makes the
transmitted sequence unbiased: over ``K`` steps against a fixed target ``t`` the
running mean of the decompressed outputs converges to ``t`` and the telescoping
sum ``sum_k q_k + residual_K == K * t`` is conserved.

Error contract
--------------
- ``TypeError`` if ``buf`` or ``residual`` is not a ``torch.float32`` tensor.
- ``ValueError`` if ``residual`` does not have the same number of elements as
  ``buf``.

Note on allowed operations
--------------------------
The framework's built-in quantization primitives are out of scope — any scale,
sign mapping, bit-level packing and the accumulator update are built explicitly
from primitive tensor ops. The current implementation below transmits the entire
buffer at full precision (no compaction at all): it is correct and unbiased, but
it moves the maximum possible number of bytes every step and leaves all of the
low-bandwidth headroom on the table.
"""

import torch

FP32 = torch.float32


def _validate(buf, residual):
    if not isinstance(buf, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError("buf and residual must be torch.Tensor")
    if buf.dtype is not FP32:
        raise TypeError(f"buf must be float32, got {buf.dtype}")
    if residual.dtype is not FP32:
        raise TypeError(f"residual must be float32, got {residual.dtype}")
    if residual.numel() != buf.numel():
        raise ValueError(
            f"residual numel {residual.numel()} must equal buf numel {buf.numel()}")


def compress(buf, residual):
    """Naive full-precision transmission (candidate start state).

    Adds the carried-over residual, then stores the whole compensated buffer at
    fp32. Because nothing is discarded the leftover residual is exactly zero and
    reconstruction is lossless — correct and unbiased, but the payload is the full
    ``4 * numel`` bytes (no bandwidth saving).
    """
    _validate(buf, residual)
    comp = buf.to(FP32) + residual.reshape(buf.shape).to(FP32)
    payload = {
        "dense": comp.reshape(-1).clone(),   # full fp32 buffer on the wire
        "numel": int(comp.numel()),
        "shape": tuple(buf.shape),
    }
    new_residual = torch.zeros_like(buf, dtype=FP32)   # lossless -> no leftover
    return payload, new_residual


def decompress(payload):
    """Reconstruct the full-shape fp32 tensor from a payload."""
    return payload["dense"].reshape(payload["shape"]).to(FP32)
