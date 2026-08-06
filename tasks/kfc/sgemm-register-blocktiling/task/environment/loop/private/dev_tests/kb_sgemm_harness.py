"""Shared harness for the fp32 SGEMM task.

Loads the candidate wrapper from /app/repo/sgemm.py (which JIT-compiles the
sibling sgemm_kernel.cu with nvcc) and the frozen baseline wrapper from
/opt/verifier-baseline/sgemm.py. Provides a high-precision float64 reference (the
ground truth — the candidate never sees it and is forbidden from calling a
prebuilt matmul), deterministic input generation scaled so the product stays
O(1), a tolerance comparison, a geometric mean, and an achieved-GFLOP/s
work-evidence proxy (the op is compute-bound, so wall-time tracks achieved fp32
throughput).
"""

import importlib.util
import math
import os

import torch

FP32 = torch.float32
FP64 = torch.float64

# Candidate-vs-reference tolerance (fp32 output vs an fp64-accumulated reference).
# Matmul is a long reduction, so fp32 reassociation vs fp64 leaves a relative gap
# that grows with K; 1e-3 comfortably covers K up to a few thousand. Validated against the oracle/baseline goldens;
# re-measure in oracle mode if you change hardware.
RTOL = 1e-3
ATOL = 1e-3


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def _load_module(path, tag):
    spec = importlib.util.spec_from_file_location("kb_sgemm_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_candidate():
    """Import the candidate wrapper (compiles /app/repo/sgemm_kernel.cu)."""
    path = os.path.join(repo_dir(), "sgemm.py")
    mod = _load_module(path, "candidate")
    if not hasattr(mod, "sgemm"):
        raise AttributeError(f"{path} does not define sgemm()")
    return mod


def load_baseline():
    """Import the frozen baseline wrapper (compiles its own sgemm_kernel.cu)."""
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/sgemm.py")
    mod = _load_module(path, "baseline")
    if not hasattr(mod, "sgemm"):
        raise AttributeError(f"{path} does not define sgemm()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation. A, B are scaled by K**-0.25 so the product
# A @ B has entries of order 1, keeping the fp32 tolerance meaningful; C is
# order 1 as well so the beta term is comparable to the product term.
# --------------------------------------------------------------------------- #
def make_matrix(shape, seed, scale=1.0, dtype=FP32, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
    return t.to(device).to(dtype)


def make_abc(M, N, K, seed, dtype=FP32):
    s = (1.0 / max(K, 1)) ** 0.25
    A = make_matrix((M, K), seed, scale=s, dtype=dtype)
    B = make_matrix((K, N), seed + 1, scale=s, dtype=dtype)
    C = make_matrix((M, N), seed + 2, scale=1.0, dtype=dtype)
    return A, B, C


# --------------------------------------------------------------------------- #
# High-precision float64 reference (the ground truth). REVIEWER-SIDE code only —
# the candidate kernel must not use a prebuilt matmul. Accumulating the product in
# float64 makes the reference independent of the candidate's reduction order.
# --------------------------------------------------------------------------- #
def ref_sgemm(A, B, C, alpha, beta):
    prod = A.to(FP64) @ B.to(FP64)
    D = alpha * prod + beta * C.to(FP64)
    return D.to(A.dtype)


def assert_close(cand, ref, name="D", msg=""):
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
# Work evidence: achieved fp32 throughput for one SGEMM (2*M*N*K FLOPs / time).
# A register-tiled solution sustains multiples of the naive one-thread-per-output
# throughput; recorded per shape as a proxy for the compute value axis.
# --------------------------------------------------------------------------- #
def achieved_gflops(M, N, K, ms):
    if ms <= 0:
        return 0.0
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
