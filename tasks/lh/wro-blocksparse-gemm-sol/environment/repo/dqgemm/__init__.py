"""dqgemm -- an int8 weight-only quantised matrix-multiply package.

Public entry: ``dq_matmul(a, b_q, scales) -> c``. Implementation lives in ``matmul.py``.
"""
from .matmul import dq_matmul

__all__ = ["dq_matmul"]
