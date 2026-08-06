#!/usr/bin/env python3
"""Standalone verifier workload for the curriculum difficulty-cluster subsystem
(scope: /app/repo/curriculum_cluster.py :: select_curriculum_cluster).

Drives the scope function on CPU (no torch, no GPU): given per-difficulty-row
sample-id buckets + row metric values, it selects the flat cluster of sample-ids for
a curriculum difficulty window, in VALUE mode (metric window) and PERCENTILE mode
(equal-count percentile band with partial boundary-row slicing).

  correctness : compare the scope's output against an INDEPENDENT reference computed
                here (NOT in the editable scope) for BOTH modes. The percentile case
                is built with a non-aligned window so the two boundary rows must be
                sliced PARTIALLY -- a whole-row shortcut selects the wrong sample-ids.
  timing      : warmup + timed repeats over a large row set with a wide percentile
                window, so the naive per-row growing concatenate dominates and
                separates from a vectorized boundary-search selection. The gap GROWS
                with the number of rows.

Emits one line ``WRO_CURRIC_RESULT {json}``. Timing uses process_time (CPU time) so
the reward band is robust to OS descheduling under fleet load.
"""
import json
import statistics
import sys
import time

import numpy as np

REPO = "/app/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# correctness: modest rows; percentile window deliberately non-bin-aligned.
C_SEED = 11
C_ROWS = 240
C_BINS = 20
C_PS = 3
C_PE = 17
# timing: many rows + wide percentile window so the naive per-row concat dominates.
T_SEED = 4
T_ROWS = 12000
T_BINS = 100
T_PS = 5
T_PE = 95
WARMUP = 1
ITERS = 3


def load_scope():
    import curriculum_cluster as m
    return m


def _gen(n_rows, seed, max_metric, min_k, max_k):
    rng = np.random.default_rng(seed)
    metric = np.sort(rng.integers(0, max_metric, size=n_rows)).astype(np.float64)
    idx_to_sample = []
    sid = 0
    for _ in range(n_rows):
        k = int(rng.integers(min_k, max_k))
        idx_to_sample.append(np.arange(sid, sid + k, dtype=np.int64))
        sid += k
    return idx_to_sample, metric


def _ref_value(index_to_sample, metric, lo, hi):
    new = None
    for row in range(len(index_to_sample)):
        if metric[row] <= hi and metric[row] > lo:
            rs = np.copy(np.asarray(index_to_sample[row]))
            new = rs if new is None else np.concatenate((new, rs), axis=None)
    return np.array([], dtype=np.int64) if new is None else new.astype(np.int64)


def _ref_percentile(index_to_sample, metric, lo, hi, num_bins):
    one_epoch = sum(len(x) for x in index_to_sample)
    per = one_epoch // num_bins
    start_count = per * lo
    end_count = one_epoch if hi == num_bins else per * hi
    new = None
    cur = 0
    for row in range(len(index_to_sample)):
        rsz = len(index_to_sample[row])
        if cur + rsz > start_count:
            rstart = max(0, start_count - cur)
            rend = rsz if cur + rsz <= end_count else end_count - cur
            rs = np.copy(np.asarray(index_to_sample[row])[rstart:rend])
            new = rs if new is None else np.concatenate((new, rs), axis=None)
        cur += rsz
        if cur >= end_count:
            break
    return np.array([], dtype=np.int64) if new is None else new.astype(np.int64)


def _eq(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _correctness_case(m):
    idx, metric = _gen(C_ROWS, C_SEED, 600, 2, 18)
    # VALUE mode over a mid metric window.
    v_lo, v_hi = 100.0, 400.0
    got_v = m.select_curriculum_cluster(idx, metric, "value", v_lo, v_hi)
    ref_v = _ref_value(idx, metric, v_lo, v_hi)
    value_ok = _eq(got_v, ref_v)
    # PERCENTILE mode, non-aligned window -> partial boundary rows.
    got_p = m.select_curriculum_cluster(idx, metric, "percentile", C_PS, C_PE, num_bins=C_BINS)
    ref_p = _ref_percentile(idx, metric, C_PS, C_PE, C_BINS)
    pct_ok = _eq(got_p, ref_p)
    # nontrivial: the percentile window must actually slice boundary rows partially
    # (else a whole-row shortcut would coincide with the reference).
    per = sum(len(x) for x in idx) // C_BINS
    start_count = per * C_PS
    cum = 0
    partial_boundary = False
    for row in range(len(idx)):
        rsz = len(idx[row])
        if cum < start_count < cum + rsz:
            partial_boundary = True
            break
        cum += rsz
    nontrivial = bool(partial_boundary and len(ref_p) > 0)
    return {"correctness_ok": bool(value_ok and pct_ok),
            "value_ok": bool(value_ok), "pct_ok": bool(pct_ok),
            "ref_value_len": int(len(ref_v)), "ref_pct_len": int(len(ref_p)),
            "nontrivial": nontrivial, "module": m.__file__}


def _timing_case(m):
    idx, metric = _gen(T_ROWS, T_SEED, T_ROWS, 1, 10)

    def once():
        m.select_curriculum_cluster(idx, metric, "percentile", T_PS, T_PE, num_bins=T_BINS)

    for _ in range(WARMUP):
        once()
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
        print("WRO_CURRIC_RESULT " + json.dumps(res))
        sys.exit(0 if res["correctness_ok"] else 3)
    elif mode == "timing":
        ms = _timing_case(m)
        print("WRO_CURRIC_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS,
            "rows": T_ROWS, "num_bins": T_BINS, "module": m.__file__}))
        sys.exit(0)
    else:
        print("WRO_CURRIC_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
