"""Shared harness for the gradient-compression-with-feedback task.

Loads the candidate from /app/repo/grad_compress.py, provides deterministic input
generation across several distributions, a trusted on-the-wire byte accountant
(the value axis is *bytes moved*: the compressed payload size), a high-precision
float32 reference for the specified per-block reconstruction, a geometric mean,
and a runtime guard that blocks the framework's quantization primitives while the
candidate runs (the packing/scale/feedback must be built from primitive ops).

The correctness invariants checked by the suite are satisfied by BOTH a lossless
full-precision compressor (the frozen baseline) AND the compact reconstruction —
what differs between them is only the payload byte count (the reward axis). A
compressor that drops the feedback residual update, or reconstructs a biased /
degenerate estimate, violates the invariants and scores zero.
"""

import contextlib
import importlib.util
import math
import os

import torch

FP32 = torch.float32

# Per-block scale granularity (elements per scale value). Disclosed in the
# behavioral contract; the reference reconstruction uses the same blocking.
BLOCK_SIZE = 2048

# Reconstruction tolerance vs the reference (per-block scale * sign). Re-measure in oracle mode if you change
# hardware. Reconstruction is scale*(+/-1); only the scale
# (an L2-norm reduction) carries any rounding slack.
RTOL = 1e-3
ATOL = 1e-4
# Error-feedback identity  new_residual == (buf + residual) - decompress  is an
# exact fp32 relation; allow only rounding-level slack.
EF_RTOL = 1e-3
EF_ATOL = 1e-3


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "grad_compress.py")
    spec = importlib.util.spec_from_file_location("candidate_grad_compress", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for sym in ("compress", "decompress"):
        if not hasattr(mod, sym):
            raise AttributeError(f"{path} does not define {sym}()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_grad(shape, seed, dist="normal", scale=1.0, device="cuda"):
    """Deterministic fp32 gradient-shaped tensor over several distributions."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    numel = 1
    for d in shape:
        numel *= d
    if dist == "zeros":
        t = torch.zeros(numel, dtype=FP32)
    elif dist == "positive":
        t = torch.rand(numel, generator=g, dtype=FP32) + 0.25
    elif dist == "negative":
        t = -(torch.rand(numel, generator=g, dtype=FP32) + 0.25)
    elif dist == "heavytail":
        # finite-variance heavy tail: standard normal cubed-ish, sign preserved
        base = torch.randn(numel, generator=g, dtype=FP32)
        t = torch.sign(base) * base.abs().pow(1.5)
    elif dist == "outlier":
        t = 0.01 * torch.randn(numel, generator=g, dtype=FP32)
        t[0] = 500.0  # a single dominating spike
    else:  # "normal"
        t = torch.randn(numel, generator=g, dtype=FP32)
    t = (t * scale).reshape(shape)
    return t.to(device).to(FP32)


# --------------------------------------------------------------------------- #
# Reference: per-block RMS scale + sign reconstruction (the ground truth).
# sign maps to {-1,+1} with sign(0) -> +1.
# --------------------------------------------------------------------------- #
def ref_reconstruct(comp, block_size=BLOCK_SIZE):
    """Return (scale_per_block, signs_flat, q_flat) for a compensated buffer.

    scale_b = ||block||_2 / sqrt(len_b); q = scale_b * sign(comp) broadcast in-block.
    The final (possibly partial) block uses its true element count.
    """
    flat = comp.reshape(-1).to(FP32)
    n = flat.numel()
    nfull = n // block_size
    rem = n - nfull * block_size
    nblocks = nfull + (1 if rem else 0)
    signs = torch.where(flat >= 0, torch.ones_like(flat), -torch.ones_like(flat))
    scale = torch.empty(nblocks, dtype=FP32, device=flat.device)
    q = torch.empty_like(flat)
    if nfull:
        full = flat[: nfull * block_size].view(nfull, block_size)
        s_full = torch.sqrt((full * full).sum(dim=1) / block_size)
        scale[:nfull] = s_full
        q[: nfull * block_size] = (
            s_full.unsqueeze(1) * signs[: nfull * block_size].view(nfull, block_size)
        ).reshape(-1)
    if rem:
        tail = flat[nfull * block_size:]
        s_tail = torch.sqrt((tail * tail).sum() / rem)
        scale[nfull] = s_tail
        q[nfull * block_size:] = s_tail * signs[nfull * block_size:]
    return scale, signs, q


# --------------------------------------------------------------------------- #
# Trusted byte accountant: the value axis (compressed payload size).
# Every torch.Tensor value contributes numel*element_size(); scalar header
# fields are counted small. A dense fp32 payload -> ~4*numel; a bit-packed
# sign payload with a per-block scale -> ~numel/8. Candidates cannot understate
# their own byte count -- this walks the real tensors they returned.
# --------------------------------------------------------------------------- #
def wire_bytes(payload):
    if isinstance(payload, dict):
        vals = list(payload.values())
    elif isinstance(payload, (tuple, list)):
        vals = list(payload)
    else:
        vals = [payload]
    total = 0
    for v in vals:
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
        elif isinstance(v, bool):
            total += 1
        elif isinstance(v, int):
            total += 4
        elif isinstance(v, float):
            total += 4
        elif isinstance(v, (tuple, list)):
            total += 4 * len(v)
    return total


def dense_bytes(numel, itemsize=4):
    """The fair 'before': transmit the whole fp32 buffer."""
    return numel * itemsize


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
def _assert_close(cand, ref, rtol, atol, name, msg=""):
    c = cand.to(FP32)
    r = ref.to(FP32)
    if tuple(c.shape) != tuple(r.shape):
        raise AssertionError(
            f"{name} shape mismatch: {tuple(c.shape)} vs {tuple(r.shape)} {msg}")
    if not torch.isfinite(c).all():
        raise AssertionError(f"{name} contains non-finite values {msg}")
    diff = (c - r).abs()
    tol = atol + rtol * r.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance "
            f"(worst excess {worst:.6f}) {msg}")


def rel_l2(a, b):
    a = a.to(FP32)
    b = b.to(FP32)
    denom = b.norm().item()
    return (a - b).norm().item() / (denom if denom > 0 else 1.0)


# --------------------------------------------------------------------------- #
# Runtime guard: block the framework quantization primitives so the candidate
# builds the scale / sign / bit-packing / feedback from primitive ops.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def forbidden_vendor_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden framework quantization primitive called during scoring")

    saved = []
    targets = [
        (torch, "quantize_per_tensor"),
        (torch, "quantize_per_channel"),
        (torch, "fake_quantize_per_tensor_affine"),
        (torch, "fake_quantize_per_channel_affine"),
    ]
    for obj, name in targets:
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            try:
                setattr(obj, name, _blocked)
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
