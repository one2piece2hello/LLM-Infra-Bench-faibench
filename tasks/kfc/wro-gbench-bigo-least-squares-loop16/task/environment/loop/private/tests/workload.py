#!/usr/bin/env python3
"""Standalone verifier workload for the Big-O complexity-estimation subsystem
(scope: /app/repo/bench_bigo.py :: compute_bigo). ACCELERATION.

Drives the scope on CPU (numpy only, no torch, no GPU). Two modes:

  correctness : build a (B, K) runtime table whose rows each follow a known
                complexity curve (with mild multiplicative noise) and check the
                returned complexity label (exact), coefficient and RMS (allclose)
                against an INDEPENDENT reference computed here (NOT part of the
                editable scope). A "select the worst-fitting curve" shortcut
                yields the wrong label and is rejected (gate vs negative).
  timing      : warmup + timed repeats over many benchmarks x curves x sizes, so
                the per-(benchmark, curve) Python least-squares loop dominates and
                separates from the vectorized batch fit. The gap grows with
                B * 6 * K.

Emits one line ``WRO_BIGO_RESULT {json}``. Timing uses process_time (CPU time) so
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

_LABELS = ["(1)", "lgN", "N", "NlgN", "N^2", "N^3"]

# correctness: modest sizes; one clear complexity per benchmark so the best-fit is
# unambiguous and the argmax shortcut is demonstrably wrong.
C_NS = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0]
C_B = 12
C_SEED = 23

# timing: many benchmarks so the per-(benchmark, curve) Python fit loop dominates.
T_B = 5000
T_K = 10
T_SEED = 7
WARMUP = 2
ITERS = 3


def load_scope():
    import bench_bigo as m
    return m


def _curve_matrix(ns):
    n = np.asarray(ns, dtype=np.float64).ravel()
    log2n = np.log2(n)
    return np.stack([np.ones_like(n), log2n, n, n * log2n, n * n, n * n * n], axis=0)


def _reference(ns, times):
    ns = np.asarray(ns, dtype=np.float64).ravel()
    t = np.asarray(times, dtype=np.float64)
    B, K = t.shape
    G = _curve_matrix(ns)
    sigma_g2 = np.square(G).sum(axis=1)
    sigma_tg = t @ G.T
    coef = sigma_tg / sigma_g2
    fit = coef[:, :, None] * G[None, :, :]
    resid = np.square(t[:, None, :] - fit).sum(axis=2)
    mean_t = t.mean(axis=1)
    rms = np.sqrt(resid / K) / mean_t[:, None]
    best = np.argmin(rms, axis=1)
    rows = np.arange(B)
    return ([_LABELS[j] for j in best], coef[rows, best], rms[rows, best])


def _gen_correctness():
    rng = np.random.default_rng(C_SEED)
    ns = np.array(C_NS, dtype=np.float64)
    G = _curve_matrix(ns)  # (6, K)
    times = np.empty((C_B, len(ns)), dtype=np.float64)
    for b in range(C_B):
        kind = 1 + (b % 5)          # cycle over lgN, N, NlgN, N^2, N^3 (skip o1)
        coef = rng.uniform(1e-3, 5.0)
        noise = 1.0 + rng.uniform(-0.03, 0.03, size=len(ns))
        times[b] = coef * G[kind] * noise
    return ns, times


def _correctness_case(m):
    ns, times = _gen_correctness()
    try:
        got = m.compute_bigo(ns, times)
    except NotImplementedError:
        return {"correctness_ok": False, "reason": "not_implemented"}
    except Exception as e:
        return {"correctness_ok": False, "reason": "exception:" + type(e).__name__}
    if not all(k in got for k in ("complexity", "coef", "rms")):
        return {"correctness_ok": False, "reason": "missing_keys"}
    ref_cx, ref_coef, ref_rms = _reference(ns, times)
    got_cx = list(got["complexity"])
    got_coef = np.asarray(got["coef"], dtype=np.float64)
    got_rms = np.asarray(got["rms"], dtype=np.float64)
    label_ok = bool(got_cx == ref_cx)
    coef_ok = bool(got_coef.shape == ref_coef.shape and np.allclose(got_coef, ref_coef, rtol=1e-7, atol=1e-12))
    rms_ok = bool(got_rms.shape == ref_rms.shape and np.allclose(got_rms, ref_rms, rtol=1e-7, atol=1e-12))
    # would an argmax (worst-fit) shortcut differ here? (gate-quality signal)
    ref_argmin_labels = ref_cx
    # recompute worst labels to confirm they differ from best
    ns2, t2 = ns, times
    G = _curve_matrix(ns2); K = ns2.shape[0]
    sg2 = np.square(G).sum(1); stg = t2 @ G.T; cf = stg / sg2
    ft = cf[:, :, None] * G[None, :, :]
    rm = np.sqrt(np.square(t2[:, None, :] - ft).sum(2) / K) / t2.mean(1)[:, None]
    worst = [_LABELS[j] for j in np.argmax(rm, axis=1)]
    nontrivial = int(worst != ref_argmin_labels)
    return {"correctness_ok": bool(label_ok and coef_ok and rms_ok),
            "label_ok": label_ok, "coef_ok": coef_ok, "rms_ok": rms_ok,
            "nontrivial": nontrivial}


def _timing_case(m):
    rng = np.random.default_rng(T_SEED)
    ns = np.geomspace(16.0, 16384.0, T_K).astype(np.float64)
    times = rng.uniform(0.5, 100.0, size=(T_B, T_K)).astype(np.float64)

    def once():
        m.compute_bigo(ns, times)

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
        print("WRO_BIGO_RESULT " + json.dumps(res))
        sys.exit(0 if res.get("correctness_ok") else 3)
    elif mode == "timing":
        ms = _timing_case(m)
        if ms < 0:
            print("WRO_BIGO_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "reason": "not_implemented"}))
            sys.exit(3)
        print("WRO_BIGO_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS, "B": T_B, "K": T_K}))
        sys.exit(0)
    else:
        print("WRO_BIGO_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
