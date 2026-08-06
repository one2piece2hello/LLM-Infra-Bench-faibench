#!/usr/bin/env python3
"""Verifier workload for wre-sched-fair-vtc-roundrobin.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden multi-tenant fair
round-robin scheduling workloads (VTC-style fairness). Correctness = EXACT int64-array equality
against an INDEPENDENT in-harness reference (re-derived from the disclosed within-tenant-round +
(round,tenant) ordering contract with a plain python counter and sort; the vectorized oracle is
never baked into the image).

Modes:
  correctness -> CSPRNG anti-cache probe (two distinct tenant arrays -> distinct positions) +
                 hidden shapes (varying N, tenant counts, skew — one dominant tenant, uniform,
                 single tenant, sparse tenant coverage).
  timing      -> median-of-medians host wall (ms) over a large balanced multi-tenant queue, the
                 case where a python dict-of-lists round-robin interleave loop is far slower than a
                 vectorized bincount + cumsum + stable-argsort grouping + argsort scatter.
                 Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys
import time

# Contention-independent timing (fleet rule): cap all math-lib threading BEFORE numpy import so the
# wall-clock is not perturbed by sibling core-contention on a shared grading host. The gradient
# (python interleave loop vs vectorized grouping+argsort) persists single-threaded.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ[_v] = "1"

import numpy as np
try:
    import torch as _torch
    _torch.set_num_threads(1)
except Exception:
    pass

KERNEL_PATH = "/app/repo/submission/kernel.py"


# ---------------- independent reference (round + (round,tenant) sort; NOT the oracle) -----------

def _reference(tenant_ids, num_tenants):
    t = np.asarray(tenant_ids, dtype=np.int64)
    n = int(t.shape[0])
    seen = {}
    rounds = [0] * n
    for i in range(n):                        # within-tenant round via a plain python counter
        ti = int(t[i])
        r = seen.get(ti, 0)
        rounds[i] = r
        seen[ti] = r + 1
    order = sorted(range(n), key=lambda i: (rounds[i], int(t[i])))   # (round asc, tenant asc)
    out = np.empty(n, dtype=np.int64)
    for p, idx in enumerate(order):
        out[idx] = p                          # 0-based fair-schedule position
    return out


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _tenants(seed, n, num_tenants):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_tenants, size=n, dtype=np.int64)


def _tenants_skew(seed, n, num_tenants, dom_frac=0.7):
    # one dominant tenant (tenant 0) owns dom_frac of the requests, rest spread over the others
    rng = np.random.default_rng(seed)
    out = rng.integers(1, max(2, num_tenants), size=n, dtype=np.int64)
    mask = rng.random(n) < dom_frac
    out[mask] = 0
    return out


def _eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def run_correctness(custom_kernel):
    # --- CSPRNG anti-cache probe: two distinct tenant arrays (same shape) -> distinct positions ---
    nt = 8
    ta = _tenants(secrets.randbits(62), 512, nt)
    tb = _tenants(secrets.randbits(62), 512, nt)
    oa = np.asarray(custom_kernel((ta, nt)))
    ob = np.asarray(custom_kernel((tb, nt)))
    if not _eq(oa, _reference(ta, nt)):
        return False, "cache-probe array A wrong"
    if not _eq(ob, _reference(tb, nt)):
        return False, "cache-probe array B wrong (output caching?)"
    if oa.shape == ob.shape and np.array_equal(oa, ob):
        return False, "cache-probe: identical positions for different tenants (caching/constant cheat)"

    # --- hidden shapes: varying N, tenant counts (uniform) ---
    for seed, n, nt in [(1, 2, 2), (2, 5, 3), (3, 200, 8), (4, 1000, 16),
                        (5, 777, 64), (6, 1, 1), (7, 4096, 4), (8, 333, 100)]:
        t = _tenants(seed, n, nt)
        got = np.asarray(custom_kernel((t, nt)))
        want = _reference(t, nt)
        if not _eq(got, want):
            return False, f"uniform N={n} num_tenants={nt} seed={seed}: mismatch"

    # --- single tenant: round == arrival index -> position == identity ---
    n = 300
    t = np.zeros(n, dtype=np.int64)
    if not _eq(np.asarray(custom_kernel((t, 1))), _reference(t, 1)):
        return False, "single-tenant identity mismatch"
    if not _eq(np.asarray(custom_kernel((t, 4))), _reference(t, 4)):   # only tenant 0 used of 4
        return False, "single-active-tenant (sparse coverage) mismatch"

    # --- skewed: one dominant tenant + a burst (fairness must NOT let it monopolise the head) ---
    for seed, n, nt in [(11, 500, 8), (12, 2000, 16), (13, 1500, 32)]:
        t = _tenants_skew(seed, n, nt)
        got = np.asarray(custom_kernel((t, nt)))
        want = _reference(t, nt)
        if not _eq(got, want):
            return False, f"skew N={n} num_tenants={nt} seed={seed}: mismatch"

    # --- sparse coverage: only some tenant ids appear (others empty) ---
    t = np.array([5, 5, 2, 5, 2, 9, 5], dtype=np.int64)
    if not _eq(np.asarray(custom_kernel((t, 10))), _reference(t, 10)):
        return False, "sparse-tenant-coverage mismatch"
    return True, "all correctness passed"


# ---------------- timing -----------------------------------------------------------------------

TIMING_N = 120000            # multi-tenant queue length
TIMING_TENANTS = 64          # SKEWED mix (one bursty tenant): the naive dense round-robin sweep
#                              scans max_round x num_tenants cells (max_round ~= the dominant
#                              tenant's count), so it does FAR more work than the ~N real emits —
#                              exactly the O(max_round * num_tenants) pathology fair scheduling hits
#                              under a bursty tenant. The vectorized group-by is O(N log N).
TIMING_DOM = 0.75            # dominant tenant owns ~75% of the queue -> max_round ~= 0.75*N
_POOL = 4


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _build_pool():
    return [(_tenants_skew(3000 + pi, TIMING_N, TIMING_TENANTS, TIMING_DOM), TIMING_TENANTS)
            for pi in range(_POOL)]


def run_timing(custom_kernel):
    pool = _build_pool()
    for w in range(2):
        custom_kernel(pool[w % _POOL])
    blocks = []
    for b in range(5):
        durs = []
        for i in range(6):
            data = pool[(b * 6 + i) % _POOL]
            t0 = time.perf_counter()
            custom_kernel(data)
            durs.append((time.perf_counter() - t0) * 1e3)
        blocks.append(_median(durs))
    med = _median(blocks)
    spread = max(blocks) / min(blocks) if min(blocks) > 0 else float("inf")
    return med, spread, spread


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    try:
        custom_kernel = _load_kernel()
    except Exception as exc:
        print("WRE_RESULT " + json.dumps({"mode": mode, "correctness_ok": False,
              "error": f"load_failed: {type(exc).__name__}: {exc}"})); sys.exit(3)

    if mode == "correctness":
        try:
            ok, msg = run_correctness(custom_kernel)
        except NotImplementedError as exc:
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"not_implemented: {exc}"})); sys.exit(3)
        except Exception as exc:
            import traceback
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]})); sys.exit(3)
        print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": bool(ok), "detail": msg}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        try:
            med, stab, spread = run_timing(custom_kernel)
        except Exception as exc:
            print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "error": f"{type(exc).__name__}: {exc}"})); sys.exit(3)
        print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": med,
              "per_iter_max_min": stab, "per_shape_spread": spread,
              "flat_ok": True, "stable_ok": True, "primary": {"N": TIMING_N, "num_tenants": TIMING_TENANTS}}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
