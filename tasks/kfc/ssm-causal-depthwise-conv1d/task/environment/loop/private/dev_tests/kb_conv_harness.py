"""Shared harness for the per-channel trailing-window weighted-sum task
(causal depthwise short-conv + bias + SiLU, sourced from mamba.py — reviewer note).

Loads the candidate from /app/repo/channel_window_op.py, provides a high-precision
float32 reference (an independent explicit shift-accumulate, NOT a library conv),
deterministic input generation, a dtype-keyed tolerance comparison, an analytic
HBM-bytes-moved work-evidence proxy, a geometric mean, and a runtime guard that
blocks the framework's 1-D convolution primitives while the candidate runs (the
operation must be built explicitly, not delegated to conv1d/cuDNN).
"""

import contextlib
import importlib.util
import math
import os

import torch

FP32 = torch.float32
BF16 = torch.bfloat16
FP16 = torch.float16


# Candidate-vs-reference tolerance, keyed by dtype. fp32 is tight; bf16/fp16 allow
# the rounding slack of a K-tap weighted sum + SiLU. Re-measure in oracle mode.
def tol_for(dtype):
    if dtype == FP32:
        return 1e-4, 1e-4
    return 2e-2, 2e-2


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "channel_window_op.py")
    spec = importlib.util.spec_from_file_location("candidate_channel_window_op", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "channel_window_op"):
        raise AttributeError(f"{path} does not define channel_window_op()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_x(B, C, L, seed, dtype=BF16, scale=1.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(B, C, L, generator=g, dtype=torch.float32) * scale
    return t.to(device).to(dtype)


def make_w(C, K, seed, dtype=BF16, device="cuda"):
    """(C, K) per-row weights; small magnitude like a trained short filter."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = 0.5 * torch.randn(C, K, generator=g, dtype=torch.float32)
    return w.to(device).to(dtype)


def make_bias(C, seed, dtype=BF16, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    b = 0.1 * torch.randn(C, generator=g, dtype=torch.float32)
    return b.to(device).to(dtype)


# --------------------------------------------------------------------------- #
# High-precision float32 reference (the ground truth).
# Independent explicit shift-accumulate over the trailing window (no conv1d), so
# it is safe to run inside forbidden_vendor_guard().
# --------------------------------------------------------------------------- #
def ref_channel_window_op(x, w, bias):
    B, C, L = x.shape
    K = int(w.shape[1])
    xf = x.to(torch.float32)
    wf = w.to(torch.float32)
    xp = torch.nn.functional.pad(xf, (K - 1, 0))          # left zero-pad by K-1
    acc = torch.zeros((B, C, L), dtype=torch.float32, device=x.device)
    for j in range(K):
        acc = acc + wf[:, j].view(1, C, 1) * xp[:, :, j:j + L]
    if bias is not None:
        acc = acc + bias.to(torch.float32).view(1, C, 1)
    y = acc * torch.sigmoid(acc)                          # SiLU gating
    return y.to(x.dtype)


def _assert_one(candidate, reference, rtol, atol, name, msg=""):
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(candidate.shape)} vs reference {tuple(reference.shape)} {msg}")
    if candidate.dtype != reference.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {candidate.dtype} vs reference {reference.dtype} {msg}")
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


def assert_close(cand, ref, dtype, msg=""):
    """Compare a single output tensor against the reference at dtype-keyed tol."""
    rtol, atol = tol_for(dtype)
    _assert_one(cand, ref, rtol, atol, "y", msg)


# --------------------------------------------------------------------------- #
# Work evidence: analytic HBM bytes moved for one (B, C, L) call.
# The fused ideal reads x (B*C*L), w (C*K), bias (C) once and writes y (B*C*L):
#   ~ (B*C*L + C*K + C) reads + (B*C*L) writes elements. The naive multi-pass
# baseline moves ~K x-passes + K acc writes + one activation pass more. Recorded
# as a portable proxy so the value axis discriminates even off the target card.
# --------------------------------------------------------------------------- #
def ideal_bytes_moved(B, C, L, K, itemsize):
    reads = (B * C * L + C * K + C) * itemsize
    writes = (B * C * L) * itemsize
    return reads + writes


# --------------------------------------------------------------------------- #
# Runtime guard: block the framework's 1-D convolution primitives so the operation
# cannot be delegated to conv1d / cuDNN. The reference uses only pad + arithmetic,
# so it is unaffected by this guard.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def forbidden_vendor_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden 1-D convolution primitive called during scoring")

    saved = []
    targets = [(torch.nn.functional, "conv1d"), (torch, "conv1d")]
    for obj, name in targets:
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            try:
                setattr(obj, name, _blocked)
            except Exception:
                saved.pop()
    # torch.nn.Conv1d module: block its forward if present
    if hasattr(torch.nn, "Conv1d"):
        cls = torch.nn.Conv1d
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
