#!/usr/bin/env python3
"""Standalone workload for the adaptive-moment optimizer subsystem in
`torch_optimizer/yogi.py`.

Drives the PUBLIC entry (the `Yogi` optimizer class) over a fixed-seed set of
MANY small parameters on the GPU. Two modes:

  correctness : run the optimizer for a fixed number of steps on fixed-seed
                parameters + gradients, and compare the resulting parameters
                against an INDEPENDENT fp32 reference optimizer computed here
                (this reference is NOT part of the editable scope), by
                relative-norm tolerance.
  timing      : warmup + timed repeats of `optimizer.step()` only (paired
                measurement done by the verifier across candidate/baseline).

Emits one line `WRO_GDN_RESULT {json}`.

The timed regime uses MANY parameters so that a per-parameter Python loop
separates clearly from a batched formulation, yet finishes in seconds.
"""
import json
import math
import os
import sys
import time

import torch

REPO = os.environ.get("WRO_REPO", "/app/repo")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from torch_optimizer import Yogi  # noqa: E402

# ---- workload configuration (the timed regime) ----
NPARAMS = 512      # number of parameters
D = 48             # each parameter is [D, D]
LR = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-3
INIT_ACC = 1e-6
KSTEPS = 8         # optimizer steps for the correctness check
WARMUP = 3
ITERS = 10
REL_MAX_TOL = 5.0e-3
REL_L2_TOL = 1.0e-3
DTYPE = torch.float32


def _make(seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    params = [torch.randn(D, D, device=device, dtype=DTYPE, generator=g) * 0.1
              for _ in range(NPARAMS)]
    grads = [torch.randn(D, D, device=device, dtype=DTYPE, generator=g) * 0.05
             for _ in range(NPARAMS)]
    return params, grads


def _yogi_reference(params0, grads, ksteps):
    """Independent trusted fp32 Yogi reference (ground truth; NOT in the editable
    scope). Mirrors the additive sign-based second-moment update, per parameter."""
    b1, b2 = BETAS
    ps = [p.clone() for p in params0]
    ms = [torch.full_like(p, INIT_ACC) for p in params0]
    vs = [torch.full_like(p, INIT_ACC) for p in params0]
    for step in range(1, ksteps + 1):
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        for i in range(len(ps)):
            g = grads[i]
            ms[i] = ms[i] * b1 + g * (1 - b1)
            gsq = g * g
            vs[i] = vs[i] + torch.sign(vs[i] - gsq) * gsq * (-(1 - b2))
            denom = (vs[i].sqrt() / math.sqrt(bc2)) + EPS
            ps[i] = ps[i] - (LR / bc1) * (ms[i] / denom)
    return ps


def _build_optim(params0, grads):
    params = [p.clone().requires_grad_(True) for p in params0]
    for p, g in zip(params, grads):
        p.grad = g.clone()
    opt = Yogi(params, lr=LR, betas=BETAS, eps=EPS,
               initial_accumulator=INIT_ACC, weight_decay=0.0)
    return params, opt


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_GDN_RESULT " + json.dumps({"error": "no_cuda"}))
        sys.exit(2)
    torch.cuda.synchronize()
    params0, grads = _make(seed=0)

    if mode == "correctness":
        params, opt = _build_optim(params0, grads)
        for _ in range(KSTEPS):
            opt.step()
        torch.cuda.synchronize()
        ref = _yogi_reference(params0, grads, KSTEPS)
        out = torch.stack([p.detach() for p in params]).float()
        rf = torch.stack(ref).float()
        diff = (out - rf).abs()
        rel_max = float(diff.max()) / float(rf.abs().max().clamp_min(1e-6))
        rel_l2 = float(diff.norm()) / float(rf.norm().clamp_min(1e-6))
        ok = (rel_max <= REL_MAX_TOL) and (rel_l2 <= REL_L2_TOL)
        print("WRO_GDN_RESULT " + json.dumps(
            {"mode": "correctness", "correctness_ok": bool(ok),
             "rel_max": rel_max, "rel_l2": rel_l2, "nparams": NPARAMS, "d": D}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        params, opt = _build_optim(params0, grads)
        for _ in range(WARMUP):
            opt.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            opt.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print("WRO_GDN_RESULT " + json.dumps(
            {"mode": "timing", "timing_ms": ms, "iters": ITERS,
             "nparams": NPARAMS, "d": D}))
        sys.exit(0)
    else:
        print("WRO_GDN_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
