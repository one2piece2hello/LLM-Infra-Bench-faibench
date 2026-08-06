"""Shared harness for the gated running-state sequence-mixer task.

Loads the candidate from ``/app/repo/gated_state_recurrence.py``, provides a
high-precision float64 sequential reference for the gated state recurrence (the
ground truth for both the output and the final state), deterministic input
generation, tolerance comparison for the ``(o[, final_state])`` outputs, an
analytic work-evidence proxy, a geometric mean, and a runtime guard that blocks
third-party sequence-mixing / attention library entry points while the candidate
runs (the recurrence must be built from primitive ops).
"""

import contextlib
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
FP16 = torch.float16

# Candidate-vs-reference tolerance. The candidate accumulates the state in fp32
# and returns a bf16/fp16 output; a chunk-parallel reformulation factors the
# per-feature decay across a block, which is slightly less exact -> allow a modest
# tolerance. Tolerance fp32 3e-3 / bf16 3e-2; re-measure in oracle mode from the
# oracle's real error.
RTOL = 3e-2
ATOL = 3e-2
# final-state parity (fp32 accumulate compared in fp32).
STATE_RTOL = 3e-2
STATE_ATOL = 3e-2


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "gated_state_recurrence.py")
    spec = importlib.util.spec_from_file_location("candidate_gated_state_recurrence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "gated_state_recurrence"):
        raise AttributeError(f"{path} does not define gated_state_recurrence()")
    return mod


# --------------------------------------------------------------------------- #
# Deterministic input generation
# --------------------------------------------------------------------------- #
def make_qkv(B, H, L, Dk, Dv, seed, dtype=BF16, device="cuda",
             gate_mode="rand", q_scale=1.0, v_scale=1.0):
    """Return (q, k, v, g) on ``device`` in ``dtype``.

    ``gate_mode`` controls the per-feature log-gate ``g`` (decay = exp(g)):
      "rand"   -> log-sigmoid of random logits (mild, decaying; typical regime),
      "zero"   -> g == 0 (no decay: the state accumulates like a plain running sum),
      "strong" -> g == -4 (near-reset each step; state barely carries forward),
      "mixed"  -> wider-spread log-sigmoid (some features keep, some forget fast).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, L, Dk, generator=gen, dtype=torch.float32) * q_scale
    k = torch.randn(B, H, L, Dk, generator=gen, dtype=torch.float32)
    v = torch.randn(B, H, L, Dv, generator=gen, dtype=torch.float32) * v_scale
    if gate_mode == "rand":
        g = F.logsigmoid(torch.randn(B, H, L, Dk, generator=gen, dtype=torch.float32))
    elif gate_mode == "zero":
        g = torch.zeros(B, H, L, Dk, dtype=torch.float32)
    elif gate_mode == "strong":
        g = torch.full((B, H, L, Dk), -4.0, dtype=torch.float32)
    elif gate_mode == "mixed":
        g = F.logsigmoid(1.5 * torch.randn(B, H, L, Dk, generator=gen, dtype=torch.float32) - 0.5)
    else:
        raise ValueError(gate_mode)
    to = lambda t: t.to(device).to(dtype)
    return to(q), to(k), to(v), to(g)


def make_state(B, H, Dk, Dv, seed, device="cuda"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(B, H, Dk, Dv, generator=gen, dtype=torch.float32) * 0.1).to(device)


# --------------------------------------------------------------------------- #
# High-precision float64 sequential reference (ground truth) for BOTH outputs.
# Mirrors the contract exactly: decay the carried state by exp(g_t), add the
# key/value outer product, then the query reads the *updated* state. Causal.
# --------------------------------------------------------------------------- #
def ref_gated_state_recurrence(q, k, v, g, initial_state=None, output_final_state=False):
    orig_dtype = q.dtype
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    qf = q.to(torch.float64) * (Dk ** -0.5)
    kf = k.to(torch.float64)
    vf = v.to(torch.float64)
    af = g.to(torch.float64).exp()
    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float64, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(torch.float64)
    o = torch.empty(B, H, L, Dv, dtype=torch.float64, device=q.device)
    for t in range(L):
        a_t = af[:, :, t].unsqueeze(-1)
        S = a_t * S + kf[:, :, t].unsqueeze(-1) * vf[:, :, t].unsqueeze(-2)
        o[:, :, t] = (qf[:, :, t].unsqueeze(-1) * S).sum(dim=-2)
    o = o.to(orig_dtype)
    if output_final_state:
        return o, S
    return o


# --------------------------------------------------------------------------- #
# Tolerance comparison
# --------------------------------------------------------------------------- #
def _assert_one(candidate, reference, rtol, atol, name, msg=""):
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"{name} shape mismatch: candidate {tuple(candidate.shape)} vs reference {tuple(reference.shape)} {msg}")
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
            f"{name}: {int(bad.sum())}/{c.numel()} elements out of tolerance (worst excess {worst:.4f}) {msg}")


def assert_output_close(cand, ref, msg="", check_dtype=True, expect_state=False):
    """``cand`` / ``ref`` are either ``o`` or ``(o, final_state)``.

    A stub that returns the wrong output (or omits the requested final state)
    must fail here."""
    if expect_state:
        if not (isinstance(cand, (tuple, list)) and len(cand) == 2):
            raise AssertionError(f"candidate must return (o, final_state), got {type(cand)} {msg}")
        co, cs = cand
        ro, rs = ref
    else:
        co = cand[0] if isinstance(cand, (tuple, list)) else cand
        ro = ref[0] if isinstance(ref, (tuple, list)) else ref
        cs = rs = None
    if check_dtype and co.dtype != ro.dtype:
        raise AssertionError(f"o dtype mismatch: candidate {co.dtype} vs reference {ro.dtype} {msg}")
    _assert_one(co, ro, RTOL, ATOL, "o", msg)
    if expect_state:
        _assert_one(cs, rs, STATE_RTOL, STATE_ATOL, "final_state", msg)


# --------------------------------------------------------------------------- #
# Work evidence: analytic multiply count for the sequential recurrence.
# Per position: decay (Dk*Dv) + key/value outer-product write (Dk*Dv) + query
# readout (Dk*Dv). A stub that drops the gate or the outer product moves less work.
# --------------------------------------------------------------------------- #
def ideal_mults(B, H, L, Dk, Dv):
    return 3 * B * H * L * Dk * Dv


# --------------------------------------------------------------------------- #
# Runtime guard: block third-party sequence-mixing / attention libraries.
# The recurrence must be built from primitive ops, not delegated to a vendor op.
# --------------------------------------------------------------------------- #
_BLOCKED_PACKAGES = ("fla", "flash_linear_attention")


class _BlockedModule:
    def __init__(self, name):
        self.__name__ = name

    def __getattr__(self, item):
        raise RuntimeError(
            f"forbidden third-party sequence-mixing library '{self.__name__}' used during scoring")


@contextlib.contextmanager
def forbidden_vendor_guard():
    saved = {}
    for name in _BLOCKED_PACKAGES:
        saved[name] = sys.modules.get(name, KeyError)
        sys.modules[name] = _BlockedModule(name)
    try:
        yield
    finally:
        for name, val in saved.items():
            if val is KeyError:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val


def geomean(values):
    vals = [max(float(v), 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
