#!/usr/bin/env python3
"""Verifier workload for ckpt-dcp-meta-bbox-merge. CPU task.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden distributed-checkpoint
shard-metadata merge workloads. Correctness = EXACT array equality against an INDEPENDENT in-harness
reference (re-derived from the disclosed contract; the oracle is not baked into the image).

Modes:
  correctness -> CSPRNG anti-cache probe (two distinct shard arrays -> distinct extents) +
                 hidden shapes (varying #tensors G, #shards N) vs the reference.
  timing      -> median-of-medians host wall (ms) over a large shard set with many tensors (the
                 case where an O(G*N) per-tensor scan is far slower than an O(N) segment-reduce).
                 Emits WRE_RESULT {"mode":"timing","timing_ms":float,...}.
"""
import json
import math
import secrets
import sys
import time

# Contention-independent timing (fleet rule): cap all math-lib threading BEFORE numpy import so
# the wall-clock is not perturbed by sibling core-contention / thread oversubscription on a shared
# scoring host. The algorithmic gradient (O(G*N) scan vs O(N) segment-reduce) persists single-threaded.
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

def _reference(entries, num_tensors):
    e = np.asarray(entries, dtype=np.int64)
    G = int(num_tensors)
    out = np.zeros(G, dtype=np.int64)
    for k in range(e.shape[0]):                    # independent O(N) per-row max reduce
        t = int(e[k, 0]); end = int(e[k, 1]) + int(e[k, 2])
        if end > out[t]:
            out[t] = end
    return out


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _gen(seed, G, shards_per):
    """Every tensor id 0..G-1 held by exactly `shards_per` shards, each a contiguous [offset,size)
    range. Rows shuffled so the max-end shard is generally NOT the first row of its tensor."""
    rng = np.random.default_rng(seed)
    tid = np.repeat(np.arange(G, dtype=np.int64), shards_per)
    offset = rng.integers(0, 1_000_000, size=tid.shape[0], dtype=np.int64)
    size = rng.integers(1, 50_000, size=tid.shape[0], dtype=np.int64)
    rows = np.stack([tid, offset, size], axis=1)
    perm = rng.permutation(rows.shape[0])
    return rows[perm], G


def _eq(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def run_correctness(custom_kernel):
    # --- CSPRNG anti-cache probe: two distinct shard arrays (same shape) -> distinct extents ---
    G, sp = 32, 3
    ea, ga = _gen(secrets.randbits(62), G, sp)
    eb, gb = _gen(secrets.randbits(62), G, sp)
    oa = np.asarray(custom_kernel((ea, ga)))
    ob = np.asarray(custom_kernel((eb, gb)))
    if not _eq(oa, _reference(ea, ga)):
        return False, "cache-probe A wrong"
    if not _eq(ob, _reference(eb, gb)):
        return False, "cache-probe B wrong (output caching?)"
    if oa.shape == ob.shape and np.array_equal(oa, ob):
        return False, "cache-probe: identical extents for different shards (caching/constant cheat)"

    # --- hidden shapes: varying #tensors G, shards-per (all >=2 exercise the max reduce) ---
    for seed, G, sp in [(1, 1, 4), (2, 3, 2), (3, 40, 3), (4, 128, 4), (5, 300, 2), (6, 7, 8)]:
        e, g = _gen(seed + 900, G, sp)
        got = np.asarray(custom_kernel((e, g)))
        want = _reference(e, g)
        if not _eq(got, want):
            return False, f"shape G={G} shards_per={sp} seed={seed}: mismatch"

    # --- explicit case: for a tensor, the largest END is NOT the largest OFFSET (size matters) ---
    #     tensor 0: shard [offset=100,size=10]->end 110 ; shard [offset=50,size=200]->end 250.
    #     max-offset shard ends at 110, but the true extent is 250 (from the smaller-offset shard).
    e = np.array([[0, 100, 10], [1, 0, 5], [0, 50, 200], [1, 20, 40]], dtype=np.int64)
    got = np.asarray(custom_kernel((e, 2)))
    want = _reference(e, 2)   # -> out[0]=250, out[1]=60
    if not _eq(got, want):
        return False, "explicit end-vs-offset (size matters) mismatch"
    return True, "all correctness passed"


# ---------------- timing -----------------------------------------------------------------------

TIMING_G = 3500          # number of tensors (drives the O(G*N) naive per-tensor scan)
TIMING_SP = 4            # shards per tensor  -> N = G*SP shard rows
_POOL = 4                # distinct pre-built inputs cycled through (anti-cache; off the clock)


def _median(v):
    s = sorted(v); n = len(s); mm = n // 2
    return s[mm] if n % 2 else (s[mm - 1] + s[mm]) / 2.0


def _build_pool():
    return [_gen(5000 + pi, TIMING_G, TIMING_SP) for pi in range(_POOL)]


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
              "flat_ok": True, "stable_ok": True, "primary": {"G": TIMING_G, "shards_per": TIMING_SP}}))
        valid = (med > 0) and math.isfinite(med)
        sys.exit(0 if valid else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
