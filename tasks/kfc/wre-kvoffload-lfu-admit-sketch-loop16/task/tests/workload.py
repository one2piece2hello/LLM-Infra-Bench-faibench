#!/usr/bin/env python3
"""Verifier workload for wre-kvoffload-lfu-admit-sketch.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden count-min-sketch KV
offload-admission queries. Correctness = EXACT array equality against an INDEPENDENT in-harness
reference (a per-block min-over-rows scan re-derived from the disclosed contract; the oracle is
never baked into the image or imported).

Modes:
  correctness -> CSPRNG anti-cache probe (two distinct sketch/key batches -> distinct admitted sets)
                 + hidden shapes (heavy row collisions so max != min, all-present, threshold above
                 all counters, D=1 and larger D, small W forcing collisions).
  timing      -> median-of-medians host wall (ms) over many blocks against a fixed sketch (the case
                 where the O(N*D) per-block python min-scan is far slower than the O(N*D) vectorized
                 gather+min). Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys
import time

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import numpy as np
try:
    import torch as _torch
    _torch.set_num_threads(1)
except Exception:
    pass

KERNEL_PATH = "/app/repo/submission/kernel.py"


# ---------------- independent reference (the disclosed contract math; NOT the oracle) ----------

def _reference(sketch, seeds, keys, present, threshold):
    sk = np.asarray(sketch)
    sd = np.asarray(seeds).astype(np.int64).tolist()
    ks = np.asarray(keys).astype(np.int64).tolist()
    pr = np.asarray(present).tolist()
    W = int(sk.shape[1]); D = int(sk.shape[0])
    rows = [sk[d].tolist() for d in range(D)]
    out = []
    for i in range(len(ks)):
        k = ks[i]
        est = min(rows[d][(k * sd[d]) % W] for d in range(D))
        if est > threshold and pr[i] == 0:
            out.append(i)
    return np.array(out, dtype=np.int64)


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


_SEEDS = np.array([1000003, 1000033, 1000037, 1000039, 1000081, 1000099], dtype=np.int64)


def _make_query(seed, N, D, W, cmax, thr, present_frac=0.1):
    rng = np.random.default_rng(seed)
    sketch = rng.integers(0, cmax + 1, size=(D, W)).astype(np.int64)
    seeds = _SEEDS[:D].copy()
    keys = rng.integers(0, 1 << 31, size=N).astype(np.int64)
    present = (rng.random(N) < present_frac).astype(np.int8)
    return (sketch, seeds, keys, present, int(thr))


def _eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def run_correctness(custom_kernel):
    # --- CSPRNG anti-cache probe: two distinct batches -> distinct admitted sets ---
    qa = _make_query(secrets.randbits(60), 2000, 4, 512, 20, 6)
    qb = _make_query(secrets.randbits(60), 2000, 4, 512, 20, 6)
    oa = np.asarray(custom_kernel(qa))
    ob = np.asarray(custom_kernel(qb))
    if not _eq(oa, _reference(*qa)):
        return False, "cache-probe batch A wrong"
    if not _eq(ob, _reference(*qb)):
        return False, "cache-probe batch B wrong (output caching?)"
    if oa.shape == ob.shape and np.array_equal(oa, ob):
        return False, "cache-probe: identical admitted sets for different sketches (caching cheat)"

    # --- hidden shapes: collisions (max != min), presence, thresholds, varying D/W ---
    cases = []
    cases.append(_make_query(1, 1500, 4, 64, 15, 5))       # small W -> heavy collisions (distinguishes max vs min)
    cases.append(_make_query(2, 1000, 1, 4096, 30, 10))    # D=1 (min == max, trivial)
    cases.append(_make_query(3, 1200, 6, 128, 25, 12))     # D=6, small W
    cases.append(_make_query(4, 800, 3, 2048, 10, 100))    # threshold above all counters -> none admitted
    cases.append(_make_query(5, 800, 3, 2048, 10, 0, present_frac=1.0))  # all present -> none admitted
    cases.append(_make_query(6, 2000, 4, 256, 18, 7))
    for seed in range(15):
        N = int(np.random.default_rng(500 + seed).integers(50, 3000))
        D = int(np.random.default_rng(600 + seed).integers(1, 7))
        W = int(np.random.default_rng(700 + seed).choice([32, 128, 512, 4096]))
        cmax = int(np.random.default_rng(800 + seed).integers(5, 40))
        thr = int(np.random.default_rng(900 + seed).integers(0, 20))
        cases.append(_make_query(1500 + seed, N, D, W, cmax, thr))
    for q in cases:
        got = np.asarray(custom_kernel(q))
        want = _reference(*q)
        if not _eq(got, want):
            return False, f"shape N={q[2].shape[0]} D={q[0].shape[0]} W={q[0].shape[1]}: mismatch"
    return True, "all correctness passed"


# ---------------- timing -----------------------------------------------------------------------

TIMING_N = 100000        # candidate blocks (drives the O(N*D) per-block python scan)
TIMING_D = 4             # count-min rows
TIMING_W = 8192          # sketch columns
TIMING_CMAX = 24
TIMING_THR = 6
_POOL = 4


def _median(v):
    s = sorted(v); m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _build_pool():
    return [_make_query(9000 + pi, TIMING_N, TIMING_D, TIMING_W, TIMING_CMAX, TIMING_THR) for pi in range(_POOL)]


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
              "flat_ok": True, "stable_ok": True, "primary": {"N": TIMING_N, "D": TIMING_D, "W": TIMING_W}}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
