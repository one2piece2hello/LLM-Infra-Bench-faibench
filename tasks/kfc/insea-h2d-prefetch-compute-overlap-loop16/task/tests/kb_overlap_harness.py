"""Shared harness for the chunked host-to-device transfer / compute-overlap task.

Loads the candidate from /app/repo/streamed_apply.py, provides a deterministic
sequential copy-then-compute reference, deterministic pinned-host chunk generation and
a device-resident per-chunk compute op, a tolerance comparison, an analytic
host-to-device bytes-moved work-evidence proxy, a geometric mean, and a runtime guard
that blocks framework auto-overlap / graph-capture conveniences while the candidate runs
(the candidate must build the copy/compute overlap explicitly with streams and events).
"""

import contextlib
import importlib.util
import math
import os

import torch

BF16 = torch.bfloat16
FP16 = torch.float16
FP32 = torch.float32

# Candidate-vs-reference tolerance. The candidate applies the SAME per-chunk `compute`
# as the reference, so the only legitimate difference is copy/compute scheduling; a
# generous bf16 tolerance still exposes any race that reads a not-yet-copied chunk.
RTOL = 5e-2
ATOL = 5e-2


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "streamed_apply.py")
    spec = importlib.util.spec_from_file_location("candidate_streamed_apply", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "streamed_chunk_apply"):
        raise AttributeError(f"{path} does not define streamed_chunk_apply()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_chunks(rows_list, D, seed, dtype=BF16, pin=True, scale=0.1):
    """A list of CPU (pinned) tensors, chunk i = (rows_list[i], D). Distinct contents
    per chunk (drawn sequentially) so a wrong chunk order / stale buffer is detectable.
    Pinned host memory lets the copy overlap compute on a side stream."""
    chunks = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for r in rows_list:
        t = (torch.randn(int(r), D, generator=g, dtype=torch.float32) * scale).to(dtype)
        if pin and r > 0 and torch.cuda.is_available():
            t = t.pin_memory()
        chunks.append(t)
    return chunks


def make_compute(D, F, seed, dtype=BF16, device="cuda"):
    """A deterministic, device-resident per-chunk GPU op: matmul against a fixed weight
    plus a bias, then a ReLU. Row-preserving: (rows, D) -> (rows, F). Opaque to the
    candidate, which must invoke it once per chunk on whatever stream it schedules."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(D, F, generator=g, dtype=torch.float32) / math.sqrt(D)).to(device).to(dtype)
    b = (0.02 * torch.randn(F, generator=g, dtype=torch.float32)).to(device).to(dtype)

    def compute(t):
        return torch.relu(torch.matmul(t, W) + b)

    return compute


# --------------------------------------------------------------------------- #
# Sequential copy-then-compute reference (the ground truth)
# --------------------------------------------------------------------------- #
def ref_streamed_apply(chunks, compute):
    if len(chunks) == 0:
        return torch.empty(0, device="cuda")
    outs = [compute(c.to("cuda", non_blocking=False)) for c in chunks]
    return torch.cat(outs, dim=0)


def assert_close(candidate, reference, msg=""):
    if not isinstance(candidate, torch.Tensor):
        raise AssertionError(f"candidate must return a torch.Tensor, got {type(candidate)} {msg}")
    if not candidate.is_cuda:
        raise AssertionError(f"candidate result must be a CUDA tensor {msg}")
    if tuple(candidate.shape) != tuple(reference.shape):
        raise AssertionError(
            f"shape mismatch: candidate {tuple(candidate.shape)} vs reference {tuple(reference.shape)} {msg}")
    if candidate.dtype != reference.dtype:
        raise AssertionError(
            f"dtype mismatch: candidate {candidate.dtype} vs reference {reference.dtype} {msg}")
    if candidate.numel() == 0:
        return
    c = candidate.to(torch.float32)
    r = reference.to(torch.float32)
    if not torch.isfinite(c).all():
        raise AssertionError(f"candidate contains non-finite values {msg}")
    diff = (c - r).abs()
    tol = ATOL + RTOL * r.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{int(bad.sum())}/{c.numel()} elements out of tolerance (worst excess {worst:.4f}) {msg}")


# --------------------------------------------------------------------------- #
# Work evidence: analytic host-to-device bytes that must be moved for one call.
# A candidate that drops a chunk (or does not actually transfer one) moves fewer
# bytes than this; recorded as a proxy alongside wall-time.
# --------------------------------------------------------------------------- #
def total_h2d_bytes(chunks):
    return int(sum(c.numel() * c.element_size() for c in chunks))


def total_rows(chunks):
    return int(sum((c.shape[0] if c.dim() >= 1 else 0) for c in chunks))


# --------------------------------------------------------------------------- #
# Runtime guard: block framework auto-overlap / graph-capture conveniences.
# Explicit torch.cuda.Stream / Event / pinned / non_blocking copies are the intended
# tools and are NOT touched here.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def forbidden_overlap_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden framework overlap / graph-capture helper called during scoring")

    saved = []
    targets = [
        (torch.cuda, "CUDAGraph"),
        (torch.cuda, "graph"),
        (torch.cuda, "make_graphed_callables"),
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
