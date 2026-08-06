"""Shared harness for the fp16 GEMM task.

Loads the candidate wrapper from /app/repo/gemm.py (which JIT-compiles the
sibling gemm_kernel.cu with nvcc) and the frozen baseline wrapper from
/opt/verifier-baseline/gemm.py. Provides a high-precision float32 matmul
reference (the ground truth — the candidate never sees it and is forbidden from
calling it), deterministic fp16 input generation scaled so C stays O(1), a
tolerance comparison, a geometric mean, and an achieved-TFLOP/s work-evidence
proxy (the op is compute-bound, so wall-time tracks achieved fp16 throughput).
"""

import importlib.util
import math
import os

import torch

FP16 = torch.float16
FP32 = torch.float32

# Candidate-vs-reference tolerance (fp16 output, float32 accumulate). Validated against the oracle/baseline goldens;
# re-measure in oracle mode if you change hardware.
RTOL = 2e-2
ATOL = 2e-2


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def _load_module(path, tag):
    spec = importlib.util.spec_from_file_location("kb_gemm_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_candidate():
    """Import the candidate wrapper (compiles /app/repo/gemm_kernel.cu)."""
    path = os.path.join(repo_dir(), "gemm.py")
    mod = _load_module(path, "candidate")
    if not hasattr(mod, "gemm"):
        raise AttributeError(f"{path} does not define gemm()")
    return mod


def load_baseline():
    """Import the frozen baseline wrapper (compiles its own gemm_kernel.cu)."""
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/gemm.py")
    mod = _load_module(path, "baseline")
    if not hasattr(mod, "gemm"):
        raise AttributeError(f"{path} does not define gemm()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation. A, B are scaled by K**-0.25 so the product
# C = A @ B has entries of order 1, keeping the fp16 tolerance meaningful.
# --------------------------------------------------------------------------- #
def make_matrix(shape, seed, scale=1.0, dtype=FP16, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
    return t.to(device).to(dtype)


def make_ab(M, N, K, seed, dtype=FP16):
    s = (1.0 / max(K, 1)) ** 0.25
    A = make_matrix((M, K), seed, scale=s, dtype=dtype)
    B = make_matrix((K, N), seed + 1, scale=s, dtype=dtype)
    return A, B


# --------------------------------------------------------------------------- #
# High-precision float32 reference (the ground truth). REVIEWER-SIDE code only —
# the candidate kernel must not use a prebuilt matmul.
# --------------------------------------------------------------------------- #
def ref_gemm(A, B):
    return (A.to(FP32) @ B.to(FP32)).to(A.dtype)


def assert_close(cand, ref, name="C", msg=""):
    if not isinstance(cand, torch.Tensor):
        raise AssertionError(f"{name}: candidate returned {type(cand)}, expected a tensor {msg}")
    if cand.shape != ref.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(cand.shape)} vs reference {tuple(ref.shape)} {msg}")
    if cand.dtype != ref.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {cand.dtype} vs reference {ref.dtype} {msg}")
    c = cand.to(FP32)
    r = ref.to(FP32)
    if not torch.isfinite(c).all():
        raise AssertionError(f"{name} contains non-finite values {msg}")
    diff = (c - r).abs()
    tol = ATOL + RTOL * r.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance "
            f"(worst excess {worst:.4f}) {msg}")


# --------------------------------------------------------------------------- #
# Work evidence: achieved fp16 throughput for one GEMM (2*M*N*K FLOPs / time).
# A tensor-core solution sustains multiples of the general FMA-path throughput;
# recorded per shape as a proxy for the compute value axis.
# --------------------------------------------------------------------------- #
def achieved_tflops(M, N, K, ms):
    if ms <= 0:
        return 0.0
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
