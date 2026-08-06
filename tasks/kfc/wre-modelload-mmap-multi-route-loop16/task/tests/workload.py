#!/usr/bin/env python3
"""Verifier workload for wre-modelload-mmap-multi-route.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden multi-file weight
name->file routing workloads. Correctness = EXACT array equality against an INDEPENDENT in-harness
reference (re-derived from the disclosed last-write-wins contract with a plain backward scan; the
vectorized oracle is never baked into the image or imported).

Modes:
  correctness -> CSPRNG anti-cache probe (two distinct query arrays -> distinct routing) + hidden
                 shapes (varying #declarations D, #names, duplicate declarations, single file).
  timing      -> median-of-medians host wall (ms) over many queries against many declarations (the
                 case where a per-query backward scan, O(Q*D), is far slower than one shared scatter
                 + gather, O(D+Q)). Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys
import time

# Contention-independent timing (fleet rule): cap all math-lib threading BEFORE numpy import.
# The algorithmic gradient (O(Q*D) per-query scan vs O(D+Q) scatter+gather) persists single-threaded.
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

def _reference(decl_name, decl_file, n_names, query):
    dn = np.asarray(decl_name, dtype=np.int64).tolist()
    df = np.asarray(decl_file, dtype=np.int64).tolist()
    D = len(dn)
    q = np.asarray(query, dtype=np.int64)
    out = np.empty(q.shape[0], dtype=np.int64)
    for i in range(q.shape[0]):
        name = int(q[i])
        found = -1
        for k in range(D - 1, -1, -1):     # backward scan: first hit == last declaration in scan order
            if dn[k] == name:
                found = df[k]
                break
        out[i] = found
    return out


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _decls(seed, D, n_names, n_files):
    """Build D declarations: each declares a random name id owned by a random file id (scan order)."""
    rng = np.random.default_rng(seed)
    # ensure every name is declared at least once, then add random extra (duplicate) declarations
    base = np.arange(n_names, dtype=np.int64)
    extra = rng.integers(0, n_names, size=max(0, D - n_names), dtype=np.int64)
    names = np.concatenate([base, extra])[:D]
    rng.shuffle(names)
    files = rng.integers(0, n_files, size=names.shape[0], dtype=np.int64)
    return names.astype(np.int64), files.astype(np.int64)


def _query(seed, Q, n_names):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_names, size=Q, dtype=np.int64)


def _eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def run_correctness(custom_kernel):
    # --- CSPRNG anti-cache probe: two distinct query arrays -> distinct routing results ---
    dn, df = _decls(7, 400, 64, 8)
    qa = _query(secrets.randbits(62), 256, 64)
    qb = _query(secrets.randbits(62), 256, 64)
    oa = np.asarray(custom_kernel((dn, df, 64, qa)))
    ob = np.asarray(custom_kernel((dn, df, 64, qb)))
    if not _eq(oa, _reference(dn, df, 64, qa)):
        return False, "cache-probe query set A wrong"
    if not _eq(ob, _reference(dn, df, 64, qb)):
        return False, "cache-probe query set B wrong (output caching?)"
    if oa.shape == ob.shape and np.array_equal(oa, ob):
        return False, "cache-probe: identical routing for different queries (caching/constant cheat)"

    # --- hidden shapes: varying #names, #declarations D (with duplicates), #files, #queries Q ---
    for seed, nn, D, nf, Q in [(1, 1, 1, 1, 20), (2, 2, 4, 2, 40), (3, 8, 40, 3, 200),
                               (4, 64, 500, 8, 1000), (5, 100, 800, 16, 777), (6, 32, 32, 4, 333)]:
        dn, df = _decls(seed, D, nn, nf)
        q = _query(seed + 500, Q, nn)
        got = np.asarray(custom_kernel((dn, df, nn, q)))
        want = _reference(dn, df, nn, q)
        if not _eq(got, want):
            return False, f"shape names={nn} D={D} files={nf} Q={Q} seed={seed}: mismatch"

    # --- last-write-wins boundary: a name declared by many files must resolve to the LAST one ---
    dn = np.array([3, 1, 3, 2, 3, 1], dtype=np.int64)     # name 3 declared by files at k=0,2,4; name 1 at k=1,5
    df = np.array([10, 11, 20, 30, 40, 50], dtype=np.int64)
    q = np.array([3, 1, 2], dtype=np.int64)               # expect [40 (last file for 3), 50 (last for 1), 30]
    got = np.asarray(custom_kernel((dn, df, 4, q)))
    want = _reference(dn, df, 4, q)
    if not _eq(got, want):
        return False, "last-write-wins boundary mismatch"
    return True, "all correctness passed"


# ---------------- timing -----------------------------------------------------------------------

TIMING_D = 4000          # number of declarations (drives the O(Q*D) per-query backward scan)
TIMING_NAMES = 2000
TIMING_FILES = 32
TIMING_Q = 4000          # number of queries
_POOL = 4                # distinct pre-built inputs cycled through (anti-cache; off the clock)


def _median(v):
    s = sorted(v); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _build_pool():
    pool = []
    for pi in range(_POOL):
        dn, df = _decls(3000 + pi, TIMING_D, TIMING_NAMES, TIMING_FILES)
        pool.append((dn, df, TIMING_NAMES, _query(4000 + pi, TIMING_Q, TIMING_NAMES)))
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
              "flat_ok": True, "stable_ok": True, "primary": {"D": TIMING_D, "Q": TIMING_Q}}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
