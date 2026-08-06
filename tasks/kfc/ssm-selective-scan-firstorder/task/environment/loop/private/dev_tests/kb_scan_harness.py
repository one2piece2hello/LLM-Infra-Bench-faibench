"""Shared harness for the first-order state-space scan task.

Loads the candidate from /app/repo/state_space_scan.py, provides a high-precision
float64 sequential reference (the ground truth), deterministic input generation, a
tolerance comparison, an analytic sequential-depth / element-count work-evidence
proxy, a geometric mean, and a runtime guard that blocks library sequence-scan
primitives while the candidate runs (the recurrence must be built explicitly).
"""

import contextlib
import importlib.util
import math
import os

import torch

F32 = torch.float32
BF16 = torch.bfloat16
FP16 = torch.float16

# Candidate-vs-reference tolerance. The reference is a float64 sequential scan; a
# correct float32 candidate differs only by reassociation / f32 rounding. Re-measure in oracle mode if you
# change hardware.
RTOL = 5e-3
ATOL = 2e-3
# bfloat16 hidden regime (state accumulated in fp32, output rounded to bf16). Re-measure in oracle mode if you change hardware.
BF16_RTOL = 5e-2
BF16_ATOL = 5e-2
# Causality / prefix slices should be (near-)identical for a deterministic causal
# algorithm; a non-causal implementation differs by O(1).
CAUSAL_RTOL = 1e-4
CAUSAL_ATOL = 1e-5


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "state_space_scan.py")
    spec = importlib.util.spec_from_file_location("candidate_state_space_scan", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "state_space_scan"):
        raise AttributeError(f"{path} does not define state_space_scan()")
    return mod


def tol_for(dtype):
    if dtype == BF16 or dtype == FP16:
        return BF16_RTOL, BF16_ATOL
    return RTOL, ATOL


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_inputs(Bsz, L, D, N, seed, dtype=F32, a_low=0.05, a_high=0.95,
                a_const=None, device="cuda"):
    """(A, B, C, x) with A in [a_low, a_high] (stable decay) unless a_const is set.

    a_const overrides A with a constant (e.g. 0.0 -> memoryless, 1.0 -> prefix-sum,
    or a value > 1 -> an amplifying / unstable recurrence)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if a_const is None:
        A = torch.rand(Bsz, L, D, N, generator=g, dtype=torch.float32) * (a_high - a_low) + a_low
    else:
        A = torch.full((Bsz, L, D, N), float(a_const), dtype=torch.float32)
    B = torch.randn(Bsz, L, N, generator=g, dtype=torch.float32)
    C = torch.randn(Bsz, L, N, generator=g, dtype=torch.float32)
    x = torch.randn(Bsz, L, D, generator=g, dtype=torch.float32)

    def to(t):
        return t.to(device).to(dtype)

    return to(A), to(B), to(C), to(x)


# --------------------------------------------------------------------------- #
# High-precision float64 sequential reference (the ground truth)
# --------------------------------------------------------------------------- #
def ref_state_space_scan(A, B, C, x):
    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)
    C64 = C.to(torch.float64)
    x64 = x.to(torch.float64)
    Bsz, L, D, N = A64.shape
    bx = x64.unsqueeze(-1) * B64.unsqueeze(2)        # (B, L, D, N)
    h = torch.zeros(Bsz, D, N, dtype=torch.float64, device=A.device)
    ys = []
    for t in range(L):
        h = A64[:, t] * h + bx[:, t]                 # (B, D, N)
        ys.append((h * C64[:, t].unsqueeze(1)).sum(dim=-1))   # (B, D)
    return torch.stack(ys, dim=1)                    # (B, L, D) float64


def assert_close(candidate, reference, rtol, atol, name="y", msg=""):
    """candidate: task-dtype tensor; reference: float64 ground truth."""
    if not isinstance(candidate, torch.Tensor):
        raise AssertionError(f"{name} must be a torch.Tensor, got {type(candidate)} {msg}")
    if tuple(candidate.shape) != tuple(reference.shape):
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(candidate.shape)} vs reference "
            f"{tuple(reference.shape)} {msg}")
    c = candidate.to(torch.float64)
    r = reference
    if not torch.isfinite(c).all() and torch.isfinite(r).all():
        raise AssertionError(f"{name} contains non-finite values where the reference is finite {msg}")
    # match non-finite pattern (amplifying / unstable recurrences may legitimately overflow)
    if not torch.isfinite(r).all():
        fin_r = torch.isfinite(r)
        if not ((torch.isfinite(c) == fin_r).all()):
            raise AssertionError(f"{name} non-finite pattern disagrees with reference {msg}")
        c = torch.where(fin_r, c, torch.zeros_like(c))
        r = torch.where(fin_r, r, torch.zeros_like(r))
    diff = (c - r).abs()
    tol = atol + rtol * r.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance "
            f"(worst excess {worst:.6f}) {msg}")


def assert_dtype_shape(y, dtype, shape, msg=""):
    if y.dtype != dtype:
        raise AssertionError(f"y dtype {y.dtype} != expected {dtype} {msg}")
    if tuple(y.shape) != tuple(shape):
        raise AssertionError(f"y shape {tuple(y.shape)} != expected {tuple(shape)} {msg}")


# --------------------------------------------------------------------------- #
# Work evidence: analytic sequential-dependency depth and element counts.
# The sequential baseline has O(L) dependent time steps; a parallel scan reduces the
# dependency depth toward O(log L). Element traffic ~ B*L*D*N per pass. Recorded as a
# proxy (the scored axis is wall-time latency, which tracks the parallel depth).
# --------------------------------------------------------------------------- #
def analytic_proxy(Bsz, L, D, N):
    return {
        "seq_depth": L,
        "log_depth": max(1, int(math.ceil(math.log2(max(L, 2))))),
        "state_elems": Bsz * L * D * N,
    }


# --------------------------------------------------------------------------- #
# Runtime guard: block library sequence-scan primitives that would delegate the
# whole recurrence in one call. The candidate must build the scan explicitly.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def forbidden_scan_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden library sequence-scan primitive called during scoring")

    saved = []
    targets = [(torch, "associative_scan")]
    hop = getattr(torch, "_higher_order_ops", None)
    if hop is not None:
        targets.append((hop, "associative_scan"))
    try:
        import triton.language as _tl  # noqa: F401
        targets.append((_tl, "associative_scan"))
    except Exception:
        pass
    for obj, attr in targets:
        if obj is not None and hasattr(obj, attr):
            saved.append((obj, attr, getattr(obj, attr)))
            try:
                setattr(obj, attr, _blocked)
            except Exception:
                saved.pop()
    try:
        yield
    finally:
        for obj, attr, val in reversed(saved):
            try:
                setattr(obj, attr, val)
            except Exception:
                pass


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
