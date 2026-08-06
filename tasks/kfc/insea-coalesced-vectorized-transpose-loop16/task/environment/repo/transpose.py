"""Fixed loader/wrapper for the 2-D transpose candidate.

DO NOT EDIT — only transpose_kernel.cu is the editable candidate file. This
module JIT-compiles the sibling transpose_kernel.cu with nvcc (torch
cpp-extension) on first use, enforces the public I/O contract, and exposes:

    transpose(x) -> y          # y[N, M] = x[M, N] transposed (y[j, i] == x[i, j])

Contract: x is a 2-D, contiguous, float32 or float16 CUDA tensor of shape
(M, N); the result y is (N, M) of the same dtype, row-major contiguous, holding
the transpose. Non-tensor or non-float32/float16 x raises TypeError; a non-2-D
or non-CUDA x raises ValueError.

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
_SRC = os.path.join(_HERE, "transpose_kernel.cu")
_TAG = hashlib.sha1(_HERE.encode("utf-8")).hexdigest()[:10]
_NAME = "insea_transpose2d_" + _TAG
_BUILD_DIR = os.path.join(tempfile.gettempdir(), "kb_transpose_build", _TAG)
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


def transpose(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype not in (torch.float32, torch.float16):
        raise TypeError("x must be a float32 or float16 tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dim() != 2:
        raise ValueError(f"x must be a 2-D matrix, got {x.dim()}-D")
    x = x.contiguous()
    return _load_ext().transpose_2d(x)
