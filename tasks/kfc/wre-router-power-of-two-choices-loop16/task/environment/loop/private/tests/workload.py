#!/usr/bin/env python3
"""Verifier workload for wre-router-power-of-two-choices.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden "power of two
choices" request-routing workloads. Correctness = EXACT array equality (both the per-request
choice array AND the final per-replica load) against an INDEPENDENT in-harness reference
(re-derived from the disclosed contract; the oracle is never baked into the image).

Modes:
  correctness -> CSPRNG anti-cache probe (two distinct candidate streams -> distinct routing) +
                 hidden shapes (varying #replicas R, batch sizes N, tie-heavy / single-replica /
                 a==b boundary cases) vs the reference.
  timing      -> median-of-medians host wall (ms) over a large stream of requests. The routing is
                 INHERENTLY SEQUENTIAL (every choice depends on all prior loads), so the honest
                 gradient is per-request python-interpreter overhead: a numpy-scalar-indexed loop
                 (baseline2) is far slower than the same loop over plain python lists (oracle).
                 Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys
import time

# Contention-independent timing (fleet rule): cap all math-lib threading BEFORE numpy import so
# the wall-clock is not perturbed by sibling core-contention / thread oversubscription on a shared
# grading host. The per-request interpreter-overhead gradient persists single-threaded.
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


# ---------------- independent reference (the disclosed contract math; NOT the oracle) ----------

def _reference(num_replicas, cand_a, cand_b, init_load):
    a = np.asarray(cand_a, dtype=np.int64)
    b = np.asarray(cand_b, dtype=np.int64)
    load = np.asarray(init_load, dtype=np.int64).copy()
    n = a.shape[0]
    choices = np.empty(n, dtype=np.int64)
    for i in range(n):
        ai = int(a[i]); bi = int(b[i])
        ci = ai if int(load[ai]) <= int(load[bi]) else bi
        choices[i] = ci
        load[ci] = load[ci] + 1
    return choices, load.astype(np.int64)


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _cands(seed, R, N):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, R, size=N, dtype=np.int64)
    b = rng.integers(0, R, size=N, dtype=np.int64)
    return a, b


def _init_load(seed, R):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, size=R, dtype=np.int64)


def _eq(got, want):
    try:
        gc, gl = got
        wc, wl = want
    except Exception:
        return False
    gc = np.asarray(gc); gl = np.asarray(gl)
    wc = np.asarray(wc); wl = np.asarray(wl)
    return (gc.shape == wc.shape and gl.shape == wl.shape
            and bool(np.array_equal(gc, wc)) and bool(np.array_equal(gl, wl)))


def run_correctness(custom_kernel):
    # --- CSPRNG anti-cache probe: two distinct candidate streams -> distinct routing ---
    R = 24
    il = _init_load(7, R)
    aa, ab = _cands(secrets.randbits(62), R, 400)
    ba, bb = _cands(secrets.randbits(62), R, 400)
    oa = custom_kernel((R, aa, ab, il))
    ob = custom_kernel((R, ba, bb, il))
    if not _eq(oa, _reference(R, aa, ab, il)):
        return False, "cache-probe stream A wrong"
    if not _eq(ob, _reference(R, ba, bb, il)):
        return False, "cache-probe stream B wrong (output caching?)"
    ca = np.asarray(oa[0]); cb = np.asarray(ob[0])
    if ca.shape == cb.shape and np.array_equal(ca, cb):
        return False, "cache-probe: identical choices for different streams (caching/constant cheat)"

    # --- hidden shapes: varying #replicas R, batch sizes N ---
    for seed, R, N in [(1, 2, 50), (2, 8, 300), (3, 32, 1500), (4, 64, 777), (5, 100, 512), (6, 5, 64)]:
        il = _init_load(seed + 900, R)
        a, b = _cands(seed, R, N)
        got = custom_kernel((R, a, b, il))
        want = _reference(R, a, b, il)
        if not _eq(got, want):
            return False, f"shape R={R} N={N} seed={seed}: mismatch"

    # --- boundary: single replica (all requests forced to source 0) ---
    R = 1
    il = np.array([3], dtype=np.int64)
    a = np.zeros(40, dtype=np.int64); b = np.zeros(40, dtype=np.int64)
    if not _eq(custom_kernel((R, a, b, il)), _reference(R, a, b, il)):
        return False, "boundary single-replica mismatch"

    # --- boundary: tie-heavy (equal starting loads -> ties must resolve to cand_a) ---
    R = 10
    il = np.zeros(R, dtype=np.int64)
    a, b = _cands(1234, R, 200)
    if not _eq(custom_kernel((R, a, b, il)), _reference(R, a, b, il)):
        return False, "boundary tie-heavy mismatch"

    # --- boundary: cand_a == cand_b everywhere (must always pick a) ---
    R = 16
    il = _init_load(55, R)
    a, _ = _cands(77, R, 150)
    if not _eq(custom_kernel((R, a, a.copy(), il)), _reference(R, a, a.copy(), il)):
        return False, "boundary a==b mismatch"
    return True, "all correctness passed"


# ---------------- timing -----------------------------------------------------------------------

TIMING_R = 64            # number of replicas
TIMING_N = 200000        # number of sequential requests (drives the per-request loop cost)
_POOL = 4                # distinct pre-built inputs cycled through (anti-cache; built off the clock)


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _build_pool():
    pool = []
    for pi in range(_POOL):
        a, b = _cands(3000 + pi, TIMING_R, TIMING_N)
        il = _init_load(4000 + pi, TIMING_R)
        pool.append((TIMING_R, a, b, il))
    return pool


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
              "flat_ok": True, "stable_ok": True, "primary": {"R": TIMING_R, "N": TIMING_N}}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
