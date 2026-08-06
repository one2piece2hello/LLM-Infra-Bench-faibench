#!/usr/bin/env python3
"""Standalone verifier workload for the counter-finalization subsystem
(scope: /app/repo/bench_counter.py :: finalize_counters). ACCELERATION.

Drives the scope on CPU (numpy only, no torch, no GPU). Two modes:

  correctness : call finalize_counters on a curated (B, C) counter table whose
                flag set deliberately includes counters that COMBINE kInvert with
                a rate / thread / iteration transform, and check the returned
                (B, C) array against an INDEPENDENT scalar reference computed here
                (NOT part of the editable scope). Because "Invert is always last",
                an implementation that inverts out of order produces a different
                result for those combined counters and is rejected (gate vs
                negative), which is why the table is seeded with such counters.
  timing      : warmup + timed repeats over a large (B, C) table so the
                per-(benchmark, counter) Python cell loop dominates and separates
                from the vectorized masked form. The gap grows with B * C.

Emits one line ``WRO_CNT_RESULT {json}``. Timing uses process_time (CPU time) so
the reward band is robust to OS descheduling under fleet load (exp §6.52).
"""
import json
import statistics
import sys
import time

import numpy as np

REPO = "/app/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

kIsRate = 1 << 0
kAvgThreads = 1 << 1
kIsIterationInvariant = 1 << 2
kAvgIterations = 1 << 3
kInvert = 1 << 31

# Flag palette used to build tables. Deliberately rich in kInvert combinations so
# the "invert must be last" rule is exercised.
FLAG_PALETTE = [
    0,
    kIsRate,
    kAvgThreads,
    kIsIterationInvariant,
    kAvgIterations,
    kIsRate | kAvgThreads,
    kIsRate | kAvgIterations,
    kIsIterationInvariant | kAvgThreads,
    kInvert,
    kIsRate | kInvert,
    kAvgThreads | kInvert,
    kIsRate | kAvgThreads | kInvert,
    kIsRate | kIsIterationInvariant,
    kAvgIterations | kAvgThreads,
]

# correctness: small table; every palette entry appears so the invert-order and
# missing-flag mistakes are all detectable.
C_B = 6
C_C = 8
C_SEED = 17

# timing: large table so the per-cell Python loop (scalar reads + flag branches)
# dominates vs the vectorized masked form.
T_B = 6000
T_C = 20
T_SEED = 5
WARMUP = 1
ITERS = 3


def load_scope():
    import bench_counter as m
    return m


def _gen(B, C, seed, curated):
    rng = np.random.default_rng(seed)
    values = rng.uniform(1.0, 1000.0, size=(B, C))
    if curated:
        # tile the full palette across the table so all combos are present
        pal = np.array(FLAG_PALETTE, dtype=np.int64)
        flags = pal[np.arange(B * C) % pal.size].reshape(B, C)
    else:
        idx = rng.integers(0, len(FLAG_PALETTE), size=(B, C))
        flags = np.array(FLAG_PALETTE, dtype=np.int64)[idx]
    iterations = rng.integers(1000, 10_000_000, size=B).astype(np.float64)
    cpu_time = rng.uniform(1e-4, 2.0, size=B)
    num_threads = rng.choice(np.array([1.0, 2.0, 4.0, 8.0]), size=B)
    return values, flags, iterations, cpu_time, num_threads


def _reference(values, flags, iterations, cpu_time, num_threads):
    # Independent, definitional scalar implementation (NOT the editable scope).
    values = np.asarray(values, dtype=np.float64)
    flags = np.asarray(flags, dtype=np.int64)
    B, C = values.shape
    out = np.empty((B, C), dtype=np.float64)
    for b in range(B):
        it_b = float(iterations[b])
        ct_b = float(cpu_time[b])
        nt_b = float(num_threads[b])
        for c in range(C):
            v = float(values[b, c])
            f = int(flags[b, c])
            if f & kIsRate:
                v /= ct_b
            if f & kAvgThreads:
                v /= nt_b
            if f & kIsIterationInvariant:
                v *= it_b
            if f & kAvgIterations:
                v /= it_b
            if f & kInvert:
                v = 1.0 / v
            out[b, c] = v
    return out


def _invert_first(values, flags, iterations, cpu_time, num_threads):
    # The classic "invert not last" mistake, for the gate-quality signal only.
    values = np.asarray(values, dtype=np.float64)
    flags = np.asarray(flags, dtype=np.int64)
    B, C = values.shape
    out = np.empty((B, C), dtype=np.float64)
    for b in range(B):
        it_b = float(iterations[b]); ct_b = float(cpu_time[b]); nt_b = float(num_threads[b])
        for c in range(C):
            v = float(values[b, c]); f = int(flags[b, c])
            if f & kInvert:
                v = 1.0 / v
            if f & kIsRate:
                v /= ct_b
            if f & kAvgThreads:
                v /= nt_b
            if f & kIsIterationInvariant:
                v *= it_b
            if f & kAvgIterations:
                v /= it_b
            out[b, c] = v
    return out


def _correctness_case(m):
    args = _gen(C_B, C_C, C_SEED, curated=True)
    try:
        got = m.finalize_counters(*args)
    except NotImplementedError:
        return {"correctness_ok": False, "reason": "not_implemented"}
    except Exception as e:
        return {"correctness_ok": False, "reason": "exception:" + type(e).__name__}
    ref = _reference(*args)
    got = np.asarray(got, dtype=np.float64)
    shape_ok = bool(got.shape == ref.shape)
    match_ok = bool(shape_ok and np.allclose(got, ref, rtol=1e-9, atol=1e-12))
    # is the invert-order shortcut detectably wrong on this table? (gate quality)
    wrong = _invert_first(*args)
    nontrivial = int(not np.allclose(wrong, ref, rtol=1e-9, atol=1e-12))
    return {"correctness_ok": match_ok, "shape_ok": shape_ok, "nontrivial": nontrivial}


def _timing_case(m):
    args = _gen(T_B, T_C, T_SEED, curated=False)

    def once():
        m.finalize_counters(*args)

    try:
        for _ in range(WARMUP):
            once()
    except NotImplementedError:
        return -1.0
    ts = []
    for _ in range(ITERS):
        t0 = time.process_time()
        once()
        ts.append((time.process_time() - t0) * 1000.0)
    return statistics.median(ts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    m = load_scope()
    if mode == "correctness":
        res = _correctness_case(m)
        res["mode"] = "correctness"
        print("WRO_CNT_RESULT " + json.dumps(res))
        sys.exit(0 if res.get("correctness_ok") else 3)
    elif mode == "timing":
        ms = _timing_case(m)
        if ms < 0:
            print("WRO_CNT_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "reason": "not_implemented"}))
            sys.exit(3)
        print("WRO_CNT_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS, "B": T_B, "C": T_C}))
        sys.exit(0)
    else:
        print("WRO_CNT_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
