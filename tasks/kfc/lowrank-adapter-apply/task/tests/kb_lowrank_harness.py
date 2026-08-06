"""Shared harness for the frozen-linear + low-rank-correction apply task.

Loads the candidate from /app/repo/lowrank_adapter_apply.py, provides a
high-precision float32 reference for the output, deterministic input generation, a
per-dtype tolerance comparison, and analytic work-evidence proxies (correction
FLOPs and the peak intermediate bytes of the full [N, K] delta) that make the
low-rank value axis explicit alongside wall-time.

The reference computes ``y = x @ Wᵀ + scale·((x @ Aᵀ) @ Bᵀ)`` in float32 and casts
to the input dtype. This is the ground truth; candidate and baseline are scored
against it, never against each other.
"""

import importlib.util
import math
import os

import torch

BF16 = torch.bfloat16
FP32 = torch.float32

# Candidate-vs-reference tolerance per dtype. The reference runs the whole op in
# fp32 then casts to the input dtype; a bf16 candidate rounds its inputs AND (for the
# naive materialized-delta path) the merged [N,K] weight to bf16 before the K-length
# matmul, so it drifts from the fp32 reference by more than a single-matmul bf16 ulp.
# Oracle-mode recalibration (H20): the frozen bf16 baseline itself, the oracle and the
# distinct baseline2 all land within 5e-2 on every workload (worst observed excess at
# 2e-2 was ~0.013 on K up to 8192); the negative (dropped scale=2.0) is off by ~100%
# of the correction (abs error ~1.0), so 5e-2 passes correct bf16 code and still
# fails the known-bad by a wide margin. fp32 stays tight.
TOL = {
    BF16: (5e-2, 5e-2),   # (rtol, atol)
    FP32: (1e-4, 1e-4),
}


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "lowrank_adapter_apply.py")
    spec = importlib.util.spec_from_file_location("candidate_lowrank_adapter_apply", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "lowrank_adapter_apply"):
        raise AttributeError(f"{path} does not define lowrank_adapter_apply()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_tensor(shape, seed, dtype=BF16, scale=1.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
    return t.to(device).to(dtype)


def make_factors(N, K, r, seed, dtype=BF16, device="cuda"):
    """Two small factors (r, K) and (N, r). factor_a ~ N(0, 1/K) and factor_b
    small like a trained low-rank correction (kept modest so the correction does
    not swamp the base output)."""
    ga = torch.Generator(device="cpu").manual_seed(seed)
    gb = torch.Generator(device="cpu").manual_seed(seed + 1)
    a = torch.randn(r, K, generator=ga, dtype=torch.float32) * (1.0 / math.sqrt(K))
    b = torch.randn(N, r, generator=gb, dtype=torch.float32) * (1.0 / math.sqrt(max(r, 1)))
    return a.to(device).to(dtype), b.to(device).to(dtype)


def make_base_weight(N, K, seed, dtype=BF16, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(N, K, generator=g, dtype=torch.float32) * (1.0 / math.sqrt(K))
    return w.to(device).to(dtype)


# --------------------------------------------------------------------------- #
# High-precision float32 reference (the ground truth)
# --------------------------------------------------------------------------- #
def ref_lowrank_adapter_apply(x, base_weight, factor_a, factor_b, scale):
    xf = x.to(torch.float32)
    base = xf @ base_weight.to(torch.float32).transpose(-1, -2)          # [.., N]
    corr = (xf @ factor_a.to(torch.float32).transpose(-1, -2)) \
        @ factor_b.to(torch.float32).transpose(-1, -2)                   # [.., N]
    y = base + float(scale) * corr
    return y.to(x.dtype)


def _tol_for(dtype):
    return TOL.get(dtype, (5e-2, 5e-2))


def _assert_one(candidate, reference, name, msg="", dtype_for_tol=None):
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(candidate.shape)} vs reference {tuple(reference.shape)} {msg}")
    if candidate.dtype != reference.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {candidate.dtype} vs reference {reference.dtype} {msg}")
    rtol, atol = _tol_for(dtype_for_tol if dtype_for_tol is not None else candidate.dtype)
    c = candidate.to(torch.float32)
    r = reference.to(torch.float32)
    if not torch.isfinite(c).all():
        raise AssertionError(f"{name} contains non-finite values {msg}")
    diff = (c - r).abs()
    tol = atol + rtol * r.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance (worst excess {worst:.5f}) {msg}")


def assert_close(cand, ref, msg="", dtype_for_tol=None):
    """cand is the single output y; ref is the fp32-reference y cast to dtype."""
    if isinstance(cand, (tuple, list)):
        raise AssertionError(f"candidate must return a single tensor y, got {type(cand)} {msg}")
    _assert_one(cand, ref, "y", msg, dtype_for_tol=dtype_for_tol)


# --------------------------------------------------------------------------- #
# Work evidence: analytic FLOPs and peak intermediate bytes.
# The efficient low-rank apply keeps the correction at O(M*r*(K+N)) FLOPs and
# never allocates the [N, K] delta; a materialized-delta candidate pays the
# N*K*r delta-formation FLOPs AND an N*K-element peak buffer. These are recorded
# as proxies alongside wall-time (the op's value axis is FLOPs + peak memory).
# --------------------------------------------------------------------------- #
def correction_flops_lowrank(M, K, N, r):
    """FLOPs of the low-rank correction path: (x@Aᵀ) then @Bᵀ (mul+add counted)."""
    return 2 * M * K * r + 2 * M * r * N          # = 2*M*r*(K+N)


def correction_flops_materialized(M, K, N, r):
    """FLOPs to form the full [N, K] delta (factor_b @ factor_a)."""
    return 2 * N * K * r


def delta_peak_bytes(N, K, itemsize):
    """Bytes of the full [N, K] delta the low-rank path never allocates."""
    return N * K * itemsize


def base_gemm_flops(M, K, N):
    return 2 * M * K * N


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
