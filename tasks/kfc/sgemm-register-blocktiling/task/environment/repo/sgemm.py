"""Fixed loader/wrapper for the fp32 matrix-multiply candidate.

DO NOT EDIT — only ``sgemm_kernel.cu`` is the editable candidate file. This module
JIT-compiles the sibling ``sgemm_kernel.cu`` with nvcc (torch cpp-extension) on
first use, enforces the public I/O contract, and exposes:

    sgemm(A, B, C, alpha, beta) -> D      # D = alpha * (A @ B) + beta * C

Contract: ``A`` (M,K), ``B`` (K,N) and ``C`` (M,N) are 2-D, contiguous, float32
CUDA tensors; the inner dimension matches (``A.shape[1] == B.shape[0]``) and
``C`` has shape (M,N). ``alpha`` and ``beta`` are python floats. The result ``D``
is a NEW (M,N) float32 tensor; the input ``C`` is not modified. Accumulation of
the product is performed in float32.

Non-float32 / dtype-mismatched inputs raise TypeError; non-2-D operands, an inner
dimension mismatch, a wrong ``C`` shape, or non-CUDA tensors raise ValueError.

The extension name and build directory are derived from this file's own directory
so the frozen baseline copy (/opt/verifier-baseline) and the candidate (/app/repo)
compile to independent artifacts and never collide.
"""
import hashlib
import os
import tempfile

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "sgemm_kernel.cu")
_TAG = hashlib.sha1(_HERE.encode("utf-8")).hexdigest()[:10]
_NAME = "insea_fp32_sgemm_" + _TAG
_BUILD_DIR = os.path.join(tempfile.gettempdir(), "kb_sgemm_build", _TAG)
_EXT = None


def _load_ext():
    """Compile (or reuse cached) the sibling .cu. Recompiles when its contents
    change — a fresh source hash forces nvcc to rebuild in this build dir."""
    global _EXT
    if _EXT is None:
        os.makedirs(_BUILD_DIR, exist_ok=True)
        _EXT = load(
            name=_NAME,
            sources=[_SRC],
            build_directory=_BUILD_DIR,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                # H20 is sm_90 (Hopper); build agent may append -ccbin g++-11.
                "-gencode=arch=compute_90,code=sm_90",
            ],
            verbose=False,
        )
    return _EXT


def sgemm(A, B, C, alpha, beta):
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
            and isinstance(C, torch.Tensor)):
        raise TypeError("A, B and C must be torch.Tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B and C must all be float32 tensors")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B and C must be CUDA tensors")
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise ValueError("A, B and C must be 2-D matrices")
    if A.size(1) != B.size(0):
        raise ValueError(
            f"inner dimensions must match: A is {tuple(A.shape)}, B is {tuple(B.shape)}")
    M, K, N = A.size(0), A.size(1), B.size(1)
    if C.size(0) != M or C.size(1) != N:
        raise ValueError(
            f"C must have shape (M,N)=({M},{N}), got {tuple(C.shape)}")
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    return _load_ext().sgemm(A, B, C, float(alpha), float(beta))
