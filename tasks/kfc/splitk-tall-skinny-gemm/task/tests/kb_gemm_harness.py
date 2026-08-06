"""Shared harness for the tall-skinny fp32 GEMM task.

Loads the candidate wrapper from /app/repo/gemm.py (which JIT-compiles the
sibling gemm_kernel.cu with nvcc) and the frozen baseline wrapper from
/opt/verifier-baseline/gemm.py. Provides a high-precision float64 matmul
reference (the ground truth — the candidate never sees it and is forbidden from
calling a prebuilt matmul), deterministic input generation scaled so C stays
O(1), a tolerance comparison, a geometric mean, and an achieved-GFLOP/s work
evidence proxy.

Tolerance is rtol=atol=1e-3: the reference and every honest candidate accumulate
in float32, and the intended solution combines partial inner-dimension sums in a
FIXED (deterministic) order, so there is no atomics-induced reassociation drift —
1e-3 is comfortably met by a correct float32 kernel and is tight enough to reject
a kernel that silently drops part of the inner-dimension reduction.
"""

import importlib.util
import math
import os

import torch

FP32 = torch.float32
FP64 = torch.float64

# Candidate-vs-reference tolerance (float32 compute vs a float64 reference).
# Re-measure in oracle mode against the oracle/baseline goldens.
RTOL = 1e-3
ATOL = 1e-3


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
# C = A @ B has entries of order 1, keeping the tolerance meaningful even when
# the inner dimension K is very large (tall-skinny regime).
# --------------------------------------------------------------------------- #
def make_matrix(shape, seed, scale=1.0, dtype=FP32, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
    return t.to(device).to(dtype)


def make_ab(M, N, K, seed, dtype=FP32):
    s = (1.0 / max(K, 1)) ** 0.25
    A = make_matrix((M, K), seed, scale=s, dtype=dtype)
    B = make_matrix((K, N), seed + 1, scale=s, dtype=dtype)
    return A, B


# --------------------------------------------------------------------------- #
# High-precision float64 reference (the ground truth). REVIEWER-SIDE code only —
# the candidate kernel must not call a prebuilt matmul. Accumulating the long
# inner dimension in float64 makes the reference independent of any float32
# reduction order, so it fairly scores every correct float32 candidate.
# --------------------------------------------------------------------------- #
def ref_gemm(A, B):
    return (A.to(FP64) @ B.to(FP64)).to(A.dtype)


def assert_close(cand, ref, name="C", msg=""):
    if not isinstance(cand, torch.Tensor):
        raise AssertionError(f"{name}: candidate returned {type(cand)}, expected a tensor {msg}")
    if cand.shape != ref.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(cand.shape)} vs reference {tuple(ref.shape)} {msg}")
    if cand.dtype != ref.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {cand.dtype} vs reference {ref.dtype} {msg}")
    c = cand.to(FP64)
    r = ref.to(FP64)
    if not torch.isfinite(c).all():
        raise AssertionError(f"{name} contains non-finite values {msg}")
    diff = (c - r).abs()
    tol = ATOL + RTOL * r.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance "
            f"(worst excess {worst:.5f}) {msg}")


# --------------------------------------------------------------------------- #
# Work evidence: achieved throughput for one GEMM (2*M*N*K FLOPs / time). For a
# few-output / large-inner-dimension shape the frozen baseline launches only a
# handful of blocks, so it sustains a small fraction of the device peak; a
# solution that spreads the inner-dimension reduction across many blocks lifts
# both occupancy and achieved throughput while returning the same result.
# --------------------------------------------------------------------------- #
def achieved_gflops(M, N, K, ms):
    if ms <= 0:
        return 0.0
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
