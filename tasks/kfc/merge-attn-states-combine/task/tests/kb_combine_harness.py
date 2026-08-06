"""Shared harness for the partial-attention-state combine task.

Loads the candidate from /app/repo/combine_attn_states.py, provides a high-precision
float32 reference for BOTH outputs, deterministic input generation, a two-output
tolerance comparison that understands the ``-inf`` empty-row log-normalizer, an
analytic HBM-bytes-moved work-evidence proxy, a geometric mean, and a runtime guard
that blocks vendor attention-state-combine primitives while the candidate runs (the
combine must be built explicitly).
"""

import contextlib
import importlib.util
import math
import os
import sys

import torch

F32 = torch.float32
BF16 = torch.bfloat16
FP16 = torch.float16


# Candidate-vs-reference tolerance, keyed by output dtype (fp32 accumulate, output
# cast back to input dtype). Re-measure in oracle mode.
def tol_for(dtype):
    if dtype == F32:
        return (1e-4, 2e-5)   # (rtol, atol)
    if dtype == BF16:
        return (8e-3, 2e-3)
    if dtype == FP16:
        return (3e-3, 1e-3)
    return (1e-2, 1e-2)


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "combine_attn_states.py")
    spec = importlib.util.spec_from_file_location("candidate_combine_attn_states", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "combine_attn_states"):
        raise AttributeError(f"{path} does not define combine_attn_states()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_partials(N, R, D, seed, dtype=F32, out_scale=1.0, lse_scale=1.0, device="cuda"):
    """N partial outputs (N, R, D) and their log-sum-exp normalizers (N, R).

    ``lse`` is centered near 0 with ``lse_scale`` spread so the chunks carry
    genuinely different weights (a stub that ignores ``lse`` must diverge).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.randn(N, R, D, generator=g, dtype=torch.float32) * out_scale
    lse = torch.randn(N, R, generator=g, dtype=torch.float32) * lse_scale
    return out.to(device).to(dtype), lse.to(device).to(dtype)


# --------------------------------------------------------------------------- #
# High-precision float32 reference (ground truth) for BOTH outputs
# --------------------------------------------------------------------------- #
def ref_combine_attn_states(partial_out, partial_lse):
    out_f = partial_out.to(torch.float32)                     # (N, R, D)
    lse_f = partial_lse.to(torch.float32)                     # (N, R)
    m = lse_f.max(dim=0).values                               # (R,)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.exp(lse_f - m_safe.unsqueeze(0))                # (N, R)
    denom = w.sum(dim=0)                                      # (R,)
    acc = (w.unsqueeze(-1) * out_f).sum(dim=0)                # (R, D)
    finite = denom > 0
    safe = torch.where(finite, denom, torch.ones_like(denom))
    out = torch.where(finite.unsqueeze(-1), acc / safe.unsqueeze(-1), torch.zeros_like(acc))
    lse = torch.where(finite, torch.log(safe) + m_safe, torch.full_like(denom, float("-inf")))
    return out.to(partial_out.dtype), lse.to(partial_lse.dtype)


def _assert_one(candidate, reference, rtol, atol, name, msg="", allow_nonfinite=False):
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(candidate.shape)} vs reference "
            f"{tuple(reference.shape)} {msg}")
    if candidate.dtype != reference.dtype:
        raise AssertionError(
            f"{name} dtype mismatch: candidate {candidate.dtype} vs reference {reference.dtype} {msg}")
    c = candidate.to(torch.float32)
    r = reference.to(torch.float32)
    r_nonfinite = ~torch.isfinite(r)
    c_nonfinite = ~torch.isfinite(c)
    if allow_nonfinite:
        # non-finite entries (e.g. -inf for an empty-row log-normalizer) must match
        # position and value exactly.
        if not torch.equal(c_nonfinite, r_nonfinite):
            raise AssertionError(f"{name}: non-finite positions differ from reference {msg}")
        if r_nonfinite.any() and not torch.equal(c[r_nonfinite], r[r_nonfinite]):
            raise AssertionError(f"{name}: non-finite values differ from reference {msg}")
        fin = torch.isfinite(r)
        cc, rr = c[fin], r[fin]
    else:
        if c_nonfinite.any():
            raise AssertionError(f"{name} contains non-finite values {msg}")
        cc, rr = c, r
    diff = (cc - rr).abs()
    tol = atol + rtol * rr.abs()
    bad = diff > tol
    if bad.any():
        worst = float((diff - tol).max())
        raise AssertionError(
            f"{name}: {int(bad.sum())}/{cc.numel()} elements out of tolerance "
            f"(worst excess {worst:.6f}) {msg}")


def assert_pair_close(cand, ref, msg=""):
    """cand and ref are each (out, lse). Check BOTH — a stub that matches only ``out``
    (or ignores ``lse``) must fail here. ``lse`` may carry ``-inf`` for empty rows."""
    if not (isinstance(cand, (tuple, list)) and len(cand) == 2):
        raise AssertionError(f"candidate must return an (out, lse) pair, got {type(cand)} {msg}")
    co, cl = cand
    ro, rl = ref
    if not isinstance(co, torch.Tensor) or not isinstance(cl, torch.Tensor):
        raise AssertionError(f"candidate (out, lse) must both be tensors {msg}")
    ort, oat = tol_for(ro.dtype)
    lrt, lat = tol_for(rl.dtype)
    _assert_one(co, ro, ort, oat, "out", msg, allow_nonfinite=False)
    _assert_one(cl, rl, lrt, lat, "lse", msg, allow_nonfinite=True)


# --------------------------------------------------------------------------- #
# Work evidence: analytic HBM bytes moved for one combine call.
# The single-pass ideal reads the two inputs and writes the two outputs:
#   reads  = (N*R*D + N*R) * itemsize      (partial_out + partial_lse)
#   writes = (R*D + R)     * itemsize      (out + lse)
# The naive multi-pass path additionally materializes the (N, R, D) weighted-partial
# intermediate (written once, read once) — ~2*N*R*D*itemsize of avoidable traffic.
# --------------------------------------------------------------------------- #
def ideal_bytes_moved(N, R, D, itemsize):
    reads = (N * R * D + N * R) * itemsize
    writes = (R * D + R) * itemsize
    return reads + writes


def naive_extra_bytes(N, R, D, itemsize):
    return 2 * N * R * D * itemsize


# --------------------------------------------------------------------------- #
# Runtime guard: block vendor attention-state-combine primitives if importable.
# No-op when the vendor modules are absent (the image ships only torch + triton).
# --------------------------------------------------------------------------- #
_VENDOR_TARGETS = [
    ("sglang.srt.layers.attention.triton_ops.merge_state", "merge_state_triton"),
    ("flash_attn", "merge_attn_states"),
    ("flash_attn.utils.merge_attn_states", "merge_attn_states"),
    ("vllm._custom_ops", "merge_attn_states"),
]


@contextlib.contextmanager
def forbidden_vendor_guard():
    def _blocked(*args, **kwargs):
        raise RuntimeError("forbidden vendor attention-state-combine primitive called during scoring")

    saved = []
    for modname, attr in _VENDOR_TARGETS:
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, attr):
            saved.append((mod, attr, getattr(mod, attr)))
            try:
                setattr(mod, attr, _blocked)
            except Exception:
                saved.pop()
    try:
        yield
    finally:
        for mod, attr, val in reversed(saved):
            try:
                setattr(mod, attr, val)
            except Exception:
                pass


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
