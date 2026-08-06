"""Shared harness for the 2-D transpose task.

Loads the candidate wrapper from /app/repo/transpose.py (which JIT-compiles the
sibling transpose_kernel.cu with nvcc) and the frozen baseline wrapper from
/opt/verifier-baseline/transpose.py. Provides an exact reference transpose (the
ground truth — the candidate never sees it and is forbidden from delegating to a
built-in reorder primitive), deterministic input generation, a bitwise-exact
comparison (this is pure data movement — every element must match exactly), a
geometric mean, and an achieved-bandwidth work-evidence proxy (the op is
memory-bound, so wall-time tracks achieved global-memory bandwidth).

This module is REVIEWER/VERIFIER-side and is not visible to the solver during
the session (/tests is mounted only at scoring); its use
of a framework transpose to build the ground-truth reference is intentional and
has nothing to do with the candidate's implement-it-yourself fence.
"""

import importlib.util
import math
import os

import torch

FP16 = torch.float16
FP32 = torch.float32


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def _load_module(path, tag):
    spec = importlib.util.spec_from_file_location("kb_transpose_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_candidate():
    """Import the candidate wrapper (compiles /app/repo/transpose_kernel.cu)."""
    path = os.path.join(repo_dir(), "transpose.py")
    mod = _load_module(path, "candidate")
    if not hasattr(mod, "transpose"):
        raise AttributeError(f"{path} does not define transpose()")
    return mod


def load_baseline():
    """Import the frozen baseline wrapper (compiles its own transpose_kernel.cu)."""
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/transpose.py")
    mod = _load_module(path, "baseline")
    if not hasattr(mod, "transpose"):
        raise AttributeError(f"{path} does not define transpose()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation. Distinct values per position so an index/block
# bug produces a detectable mismatch (a constant fill would hide a wrong swap).
# --------------------------------------------------------------------------- #
def make_matrix(M, N, seed, dtype=FP32, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(M, N, generator=g, dtype=torch.float32)
    return t.to(device).to(dtype)


# --------------------------------------------------------------------------- #
# Exact reference transpose (the ground truth). REVIEWER-SIDE code only — the
# candidate kernel must reorder the data itself. .contiguous() materializes the
# reference in row-major (N, M) layout to compare against the candidate's
# contiguous output element-for-element.
# --------------------------------------------------------------------------- #
def ref_transpose(x):
    return x.t().contiguous()


def assert_exact(cand, ref, name="y", msg=""):
    """Bitwise-exact check — the transpose moves elements unchanged, so every
    element must match exactly (no tolerance). Shape/dtype/finiteness checked too."""
    if not isinstance(cand, torch.Tensor):
        raise AssertionError(f"{name}: candidate returned {type(cand)}, expected a tensor {msg}")
    if cand.shape != ref.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(cand.shape)} vs reference {tuple(ref.shape)} {msg}")
    if cand.dtype != ref.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {cand.dtype} vs reference {ref.dtype} {msg}")
    if not cand.is_contiguous():
        raise AssertionError(f"{name} must be row-major contiguous {msg}")
    c = cand.to(FP32)
    r = ref.to(FP32)
    if not torch.isfinite(c).all():
        raise AssertionError(f"{name} contains non-finite values {msg}")
    if not torch.equal(c, r):
        bad = (c != r)
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{c.numel()} elements differ from the reference transpose {msg}")


# --------------------------------------------------------------------------- #
# Work evidence: achieved effective bandwidth for one transpose (read all bytes
# once + write all bytes once) / time. A coalesced solution sustains a large
# multiple of the strided baseline's effective bandwidth; recorded per shape as a
# proxy for the memory-bound value axis. NOTE: the bytes moved are IDENTICAL for a
# coalesced and a strided kernel — only wall-time / achieved bandwidth
# distinguishes them, which is why the scored metric is wall-time, not bytes.
# --------------------------------------------------------------------------- #
def achieved_gbps(M, N, itemsize, ms):
    if ms <= 0:
        return 0.0
    total_bytes = 2.0 * M * N * itemsize   # read once + write once
    return total_bytes / (ms * 1e-3) / 1e9


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
