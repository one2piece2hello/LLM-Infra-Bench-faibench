"""Shared harness for the cross-entropy loss+gradient contract.

Provides:
  * ``load_candidate`` — import the candidate module from /app/repo,
  * deterministic input generation (``make_logits`` / ``make_labels``),
  * a high-precision float32 reference (``ref_ce``) — the ground truth for both
    loss and gradient,
  * ``assert_loss_close`` / ``assert_grad_close`` tolerance comparisons,
  * ``count_nonzero_rows`` — the work-evidence signal,
  * ``geomean``,
  * ``forbidden_ce_guard`` — a runtime guard that blocks the framework's
    single-call fused cross-entropy loss while the candidate runs, so the
    candidate cannot delegate the whole computation to the vendor primitive it
    is meant to build.

The reference materialises full-size buffers on purpose (it is only the
correctness oracle, not scored on memory).
"""

import contextlib
import importlib.util
import math
import os

import torch

# candidate-vs-reference tolerances
LOSS_ATOL = 1e-4
LOSS_ATOL_BF16 = 3e-2
LOSS_RTOL = 1e-3
# bf16 loss carries the dtype's own ~2^-8 (~0.4%) quantization; at a large-vocab
# loss magnitude (~19) that is ~0.075 absolute, which a 1e-3 relative term cannot
# represent. Comparing a correctly bf16-cast loss (reference/oracle) against the
# fp32 baseline loss must tolerate that quantization, so bf16 uses a relative term
# above bf16's resolution. The (N,V) gradient parity check remains the tight,
# authoritative correctness signal.
LOSS_RTOL_BF16 = 8e-3
GRAD_ATOL = 1e-4
GRAD_RTOL = 1e-3
GRAD_ATOL_BF16 = 8e-3
GRAD_RTOL_BF16 = 1e-2


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "cross_entropy_grad.py")
    spec = importlib.util.spec_from_file_location("candidate_ce_grad", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "cross_entropy_loss_grad"):
        raise AttributeError(f"{path} does not define cross_entropy_loss_grad()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_logits(N, V, seed, dtype=torch.float32, scale=4.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(N, V, generator=g, dtype=torch.float32) * scale
    return x.to(device).to(dtype)


def make_labels(N, V, seed, ignore_frac=0.0, ignore_index=-100, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lab = torch.randint(0, V, (N,), generator=g, dtype=torch.int64)
    if ignore_frac > 0.0:
        mask = torch.rand(N, generator=g) < ignore_frac
        lab[mask] = ignore_index
    return lab.to(device)


# --------------------------------------------------------------------------- #
# High-precision float32 reference (ground truth for loss + grad)
# --------------------------------------------------------------------------- #
def ref_ce(logits, labels, ignore_index=-100, label_smoothing=0.0):
    N, V = logits.shape
    s = float(label_smoothing)
    valid = labels != ignore_index
    n_valid = int(valid.sum().item())
    denom = max(n_valid, 1)
    safe = labels.clamp(min=0).long()
    idx = torch.arange(N, device=logits.device)

    lf = logits.to(torch.float32)
    logp = torch.log_softmax(lf, dim=-1)
    logp_y = logp[idx, safe]
    nll = -logp_y
    smooth = -logp.mean(dim=1)
    row_loss = (1.0 - s) * nll + s * smooth
    row_loss = torch.where(valid, row_loss, torch.zeros_like(row_loss))
    loss = row_loss.sum() / denom

    prob = logp.exp()
    grad = prob.clone()
    if s != 0.0:
        grad = grad - (s / V)
    grad[idx, safe] = grad[idx, safe] - (1.0 - s)
    grad = grad / denom
    grad[~valid] = 0.0
    return loss.to(logits.dtype), grad.to(logits.dtype)


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #
def assert_loss_close(cand_loss, ref_loss, bf16=False, msg=""):
    cl = torch.as_tensor(cand_loss).to(torch.float32).reshape(())
    rl = torch.as_tensor(ref_loss).to(torch.float32).reshape(())
    if not torch.isfinite(cl):
        raise AssertionError(f"candidate loss is non-finite {msg}")
    atol = LOSS_ATOL_BF16 if bf16 else LOSS_ATOL
    rtol = LOSS_RTOL_BF16 if bf16 else LOSS_RTOL
    d = float((cl - rl).abs())
    tol = atol + rtol * float(rl.abs())
    if d > tol:
        raise AssertionError(f"loss mismatch: |{float(cl):.6f}-{float(rl):.6f}|={d:.6f} > {tol:.6f} {msg}")


def assert_grad_close(cand_grad, ref_grad, bf16=False, msg=""):
    if tuple(cand_grad.shape) != tuple(ref_grad.shape):
        raise AssertionError(f"grad shape {tuple(cand_grad.shape)} vs {tuple(ref_grad.shape)} {msg}")
    if cand_grad.dtype != ref_grad.dtype:
        raise AssertionError(f"grad dtype {cand_grad.dtype} vs {ref_grad.dtype} {msg}")
    c = cand_grad.to(torch.float32)
    r = ref_grad.to(torch.float32)
    if not torch.isfinite(c).all():
        raise AssertionError(f"candidate grad has non-finite values {msg}")
    atol = GRAD_ATOL_BF16 if bf16 else GRAD_ATOL
    rtol = GRAD_RTOL_BF16 if bf16 else GRAD_RTOL
    diff = (c - r).abs()
    tol = atol + rtol * r.abs()
    bad = diff > tol
    if bad.any():
        raise AssertionError(
            f"{int(bad.sum())}/{c.numel()} grad elements out of tolerance "
            f"(worst excess {float((diff - tol).max()):.5f}) {msg}")


def count_nonzero_rows(grad):
    """Work-evidence: number of rows whose gradient is not all-zero."""
    return int((grad.to(torch.float32).abs().sum(dim=1) > 0).sum().item())


# --------------------------------------------------------------------------- #
# Runtime guard: block the framework's single-call fused cross-entropy loss
# while the candidate runs (the candidate must compute loss + grad itself).
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def forbidden_ce_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden fused cross-entropy primitive called during scoring")

    class _BlockedModule:
        def __init__(self, *a, **k):
            raise RuntimeError("forbidden fused cross-entropy primitive called during scoring")

    saved = []
    for obj, name, repl in [
        (torch.nn.functional, "cross_entropy", _blocked),
        (torch.nn, "CrossEntropyLoss", _BlockedModule),
    ]:
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            try:
                setattr(obj, name, repl)
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
    vals = [max(float(v), 1e-12) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
