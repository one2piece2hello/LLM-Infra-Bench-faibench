"""w8a16 — a group-quantised int8 weight / fp16 activation matmul subsystem.

Public contract (fixed; the verifier imports exactly this):

    w8a16_matmul(a, qweight, scales, zeros, group_size) -> torch.Tensor

Computes ``a @ dequant(qweight, scales, zeros, group_size)`` for a half-precision
activation and an ASYMMETRIC group-quantised int8 weight (per-group scale + integer
zero-point). See ``matmul.py`` for the exact layout and numerical contract.
"""
from .matmul import w8a16_matmul

__all__ = ["w8a16_matmul"]
