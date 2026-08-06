"""Shared harness for the scaled-dot-product causal-attention task.

Loads the candidate from /app/repo/causal_attention.py, provides a high-precision
float32 reference, deterministic input generation, a tolerance comparison, an
analytic peak-memory work-evidence proxy (the naive S*S term vs the S*D working
set), a geometric mean, and a runtime guard that blocks the framework's fused
attention primitives while the candidate runs (the attention must be built
explicitly).
"""

import contextlib
import importlib.util
import math
import os

import torch

BF16 = torch.bfloat16
FP16 = torch.float16

# Candidate-vs-reference tolerance (bf16/fp16 out, fp32 accumulate). Re-measure in oracle mode if you change
# hardware. bf16 attention output after a softmax@V carries
# ~1% rounding; fp16 is tighter.
RTOL_BF16 = 2e-2
ATOL_BF16 = 2e-2
RTOL_FP16 = 1e-2
ATOL_FP16 = 1e-2


def tols(dtype):
    if dtype == FP16:
        return RTOL_FP16, ATOL_FP16
    return RTOL_BF16, ATOL_BF16


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "causal_attention.py")
    spec = importlib.util.spec_from_file_location("candidate_causal_attention", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "causal_attention"):
        raise AttributeError(f"{path} does not define causal_attention()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_tensor(shape, seed, dtype=BF16, scale=1.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
    return t.to(device).to(dtype)


def make_qkv(B, H, Hk, S, D, seed, dtype=BF16, scale=1.0, device="cuda"):
    q = make_tensor((B, H, S, D), seed, dtype=dtype, scale=scale, device=device)
    k = make_tensor((B, Hk, S, D), seed + 1, dtype=dtype, scale=scale, device=device)
    v = make_tensor((B, Hk, S, D), seed + 2, dtype=dtype, scale=scale, device=device)
    return q, k, v


# --------------------------------------------------------------------------- #
# High-precision float32 reference (the ground truth)
# --------------------------------------------------------------------------- #
def ref_causal_attention(q, k, v, scale, causal=True):
    B, H, S, D = q.shape
    Hk = k.shape[1]
    group = H // Hk
    qf = q.to(torch.float32)
    kf = k.to(torch.float32).repeat_interleave(group, dim=1)
    vf = v.to(torch.float32).repeat_interleave(group, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * float(scale)
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)
    return out.to(q.dtype)


def assert_close(candidate, reference, dtype, name="out", msg=""):
    if not isinstance(candidate, torch.Tensor):
        raise AssertionError(f"{name} must be a torch.Tensor, got {type(candidate)} {msg}")
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(candidate.shape)} vs reference "
            f"{tuple(reference.shape)} {msg}")
    if candidate.dtype != reference.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {candidate.dtype} vs reference {reference.dtype} {msg}")
    rtol, atol = tols(dtype)
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
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance "
            f"(worst excess {worst:.4f}) {msg}")


# --------------------------------------------------------------------------- #
# Work evidence: analytic peak-memory terms (in elements) for one call.
# The naive path materializes B*H*S*S similarity + probability entries; a working
# set that never forms the (S,S) matrix stays ~ B*H*S*D. Recorded as a proxy so a
# candidate that secretly builds the full matrix cannot hide.
# --------------------------------------------------------------------------- #
def naive_score_elems(B, H, S):
    return B * H * S * S


def working_set_elems(B, H, S, D):
    return B * H * S * D


# --------------------------------------------------------------------------- #
# Runtime guard: block the framework fused attention primitives.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def forbidden_vendor_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden fused attention primitive called during scoring")

    saved = []
    targets = [
        (torch.nn.functional, "scaled_dot_product_attention"),
        (torch, "scaled_dot_product_attention"),
    ]
    for obj, name in targets:
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            try:
                setattr(obj, name, _blocked)
            except Exception:
                saved.pop()
    if hasattr(torch.nn, "MultiheadAttention"):
        cls = torch.nn.MultiheadAttention
        if hasattr(cls, "forward"):
            saved.append((cls, "forward", cls.forward))
            try:
                cls.forward = _blocked
            except Exception:
                saved.pop()
    try:
        yield
    finally:
        for obj, name, val in reversed(saved):
            try:
                setattr(obj, name, val)
            except Exception:
                pass


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
