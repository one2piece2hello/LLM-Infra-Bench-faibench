"""Fixed loader/wrapper for the fp16 GEMM candidate.

DO NOT EDIT — only gemm_kernel.cu is the editable candidate file. This module
JIT-compiles the sibling gemm_kernel.cu with nvcc (torch cpp-extension) on first
use, enforces the public I/O contract, and exposes:

    gemm(A, B) -> C            # C[M,N] = A[M,K] @ B[K,N]

Contract: A (M,K) and B (K,N) are 2-D, contiguous, fp16 CUDA tensors with a
matching inner dimension; C is (M,N) fp16, computed with float32 accumulation.
Non-fp16 / dtype-mismatched inputs raise TypeError; non-2-D / inner-mismatch /
non-CUDA shapes raise ValueError.

The extension name and build directory are derived from this file's own
directory so the frozen baseline copy (/opt/verifier-baseline) and the candidate
(/app/repo) compile to independent artifacts and never collide.
"""
import hashlib
import os
import tempfile

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "gemm_kernel.cu")
_TAG = hashlib.sha1(_HERE.encode("utf-8")).hexdigest()[:10]
_NAME = "insea_fp16_gemm_" + _TAG
_BUILD_DIR = os.path.join(tempfile.gettempdir(), "kb_gemm_build", _TAG)
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


def gemm(A, B):
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensors")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError("A and B must both be float16 (fp16) tensors")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors")
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2-D matrices")
    if A.size(1) != B.size(0):
        raise ValueError(
            f"inner dimensions must match: A is {tuple(A.shape)}, B is {tuple(B.shape)}")
    A = A.contiguous()
    B = B.contiguous()
    return _load_ext().gemm_fp16(A, B)
