#!/usr/bin/env python3
"""Verifier workload for wre-diloco-comm-chunk-balance.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden gradient-tensor size
streams. PURE HOST-LOGIC (no torch / no CUDA). Grounded in TRAIN.PARALLEL.DECENTRALIZED /
LOCAL_SGD: in DiLoCo outer-step / ring all-reduce / reduce-scatter, the flattened gradient is
transmitted as a bounded number of CONTIGUOUS chunks (parameter order); a ring all-reduce runs in
lock-step rounds, so its time is bounded by the LARGEST chunk. The solver places the chunk
boundaries (<= num_chunks contiguous chunks) to minimize the bottleneck (largest) chunk's bytes.

Metric = total bottleneck bytes (sum over the primary suite of each instance's largest-chunk byte
total) -- LOWER is better. It is emitted in the `timing_ms` field so the shared, metric-agnostic
test.sh computes
    vs_oracle = oracle_ms / candidate_ms = oracle_bottleneck / candidate_bottleneck
(a more balanced partition than the oracle -> vs_oracle > 1; equal -> 1.0; the equal-count
baseline2 -> 0 < vs_oracle < 1). The bottleneck is computed HERE from the returned boundaries + the
(hidden) sizes, so it cannot be faked; a smaller bottleneck requires a genuinely more balanced VALID
partition. Validity (boundaries strictly increasing, in 1..N, last == N, at most num_chunks chunks)
is a hard prerequisite.

Modes:
  correctness -> anti-cache probe (two different streams, both partitions VALID) + all hidden
                 streams validated. Emits WRE_RESULT {"mode":"correctness","correctness_ok":bool,...}
  timing      -> total bottleneck bytes over the primary suite. Emits
                 WRE_RESULT {"mode":"timing","timing_ms":<total_bottleneck>,...}
"""
import json
import math
import random
import secrets
import sys

KERNEL_PATH = "/app/repo/submission/kernel.py"

# Primary suite: realistic gradient size streams (a few HUGE tensors like embeddings/lm_head + many
# small norms/biases + medium linear weights), P chunks well below N -> balanced cuts beat naive.
# Metric SUMS each instance's bottleneck-chunk bytes (dominated by the largest streams).
PRIMARY_SUITE = [
    {"N": 220, "num_chunks": 8, "seed": 900},
    {"N": 300, "num_chunks": 16, "seed": 901},
    {"N": 360, "num_chunks": 12, "seed": 902},
    {"N": 420, "num_chunks": 24, "seed": 903},
    {"N": 260, "num_chunks": 8, "seed": 904},
    {"N": 340, "num_chunks": 16, "seed": 905},
]

# Hidden correctness streams -- single tensor, P=1 edge (one chunk = all), P>=N (per-tensor ok),
# odd, non-pow2, large.
CORRECTNESS_STREAMS = [
    {"N": 1, "num_chunks": 4, "seed": 101},
    {"N": 8, "num_chunks": 1, "seed": 102},      # P=1 -> a single chunk over everything
    {"N": 9, "num_chunks": 32, "seed": 103},     # P >= N -> per-tensor chunks allowed
    {"N": 37, "num_chunks": 5, "seed": 104},     # odd
    {"N": 257, "num_chunks": 16, "seed": 105},   # non-pow2
    {"N": 400, "num_chunks": 12, "seed": 106},
]

SPREAD_STREAMS = [
    {"N": 60, "num_chunks": 4, "seed": 201},
    {"N": 240, "num_chunks": 10, "seed": 202},
    {"N": 440, "num_chunks": 28, "seed": 203},
]
_PER_STREAM_SPREAD_MIN = 1.5


def _gen_sizes(spec):
    """Deterministic gradient-tensor byte stream in parameter order: a few HUGE tensors (embeddings
    / lm_head, ~10-40x a normal weight) placed at scattered positions, plus medium linear weights
    and many tiny norm/bias tensors. The skew is what makes chunk placement matter -- an equal-count
    split clusters big tensors and blows the bottleneck; balanced cuts isolate them."""
    rng = random.Random(spec["seed"])
    N = spec["N"]
    sizes = []
    for _ in range(N):
        r = rng.random()
        if r < 0.20:
            sizes.append(rng.randint(1, 30))          # tiny: norms / biases
        elif r < 0.82:
            sizes.append(rng.randint(150, 900))       # medium: linear weights
        else:
            sizes.append(rng.randint(9000, 60000))    # huge: embeddings / lm_head (~18%)
    return sizes


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _validate_and_bottleneck(boundaries, sizes, num_chunks):
    """Return (valid, bottleneck, msg). Valid iff boundaries is a non-empty strictly-increasing int
    list, every value in 1..N, last == N, len <= num_chunks. bottleneck = max chunk byte sum."""
    n = len(sizes)
    if boundaries is None:
        return False, -1, "returned None"
    try:
        b = [int(x) for x in boundaries]
    except Exception as exc:
        return False, -1, f"boundaries not int-coercible: {exc}"
    if len(b) == 0:
        return False, -1, "empty boundaries"
    if len(b) > num_chunks:
        return False, -1, f"used {len(b)} chunks > num_chunks {num_chunks}"
    for k in range(len(b)):
        if b[k] < 1 or b[k] > n:
            return False, -1, f"boundary {b[k]} out of range 1..{n}"
        if k > 0 and b[k] <= b[k - 1]:
            return False, -1, f"boundaries not strictly increasing at {k}: {b[k-1]} then {b[k]}"
    if b[-1] != n:
        return False, -1, f"last boundary {b[-1]} != N {n} (not all tensors covered)"
    bottleneck = 0
    start = 0
    for end in b:
        s = sum(sizes[start:end])
        if s > bottleneck:
            bottleneck = s
        start = end
    return True, bottleneck, "valid"


def _plan(custom_kernel, spec):
    sizes = _gen_sizes(spec)
    data = (list(sizes), spec["num_chunks"], {"N": len(sizes), "num_chunks": spec["num_chunks"]})
    out = custom_kernel(data)
    valid, bn, msg = _validate_and_bottleneck(out, sizes, spec["num_chunks"])
    return valid, bn, msg


def run_correctness(custom_kernel):
    # Stage 0 -- anti-cache probe: two DIFFERENT streams (different N), both partitions VALID.
    for _ in range(2):
        spec = {"N": random.Random(secrets.randbits(28)).randint(50, 150),
                "num_chunks": 8, "seed": secrets.randbits(28) | 1}
        valid, _, msg = _plan(custom_kernel, spec)
        if not valid:
            return False, f"cache-probe stream (N={spec['N']}, seed {spec['seed']}): {msg}"
    # Stage 1 -- all hidden streams.
    for spec in CORRECTNESS_STREAMS:
        valid, _, msg = _plan(custom_kernel, spec)
        if not valid:
            return False, f"stream {spec}: {msg}"
    return True, "all correctness passed"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    try:
        custom_kernel = _load_kernel()
    except Exception as exc:
        print("WRE_RESULT " + json.dumps({"mode": mode, "correctness_ok": False,
              "error": f"load_failed: {type(exc).__name__}: {exc}"}))
        sys.exit(3)

    if mode == "correctness":
        try:
            ok, msg = run_correctness(custom_kernel)
        except NotImplementedError as exc:
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"not_implemented: {exc}"}))
            sys.exit(3)
        except Exception as exc:
            import traceback
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]}))
            sys.exit(3)
        print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": bool(ok), "detail": msg}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        try:
            total = 0
            for spec in PRIMARY_SUITE:
                valid, bn, msg = _plan(custom_kernel, spec)
                if not valid:
                    print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                          "error": f"invalid_plan on {spec['seed']}: {msg}"}))
                    sys.exit(4)
                total += bn
            spread = []
            for spec in SPREAD_STREAMS:
                v2, bn2, _ = _plan(custom_kernel, spec)
                spread.append(bn2 if (v2 and bn2 > 0) else -1)
        except Exception as exc:
            print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "error": f"{type(exc).__name__}: {exc}"}))
            sys.exit(3)
        spread_ratio = (max(spread) / min(spread)) if (spread and min(spread) > 0) else float("inf")
        flat_ok = spread_ratio >= _PER_STREAM_SPREAD_MIN
        print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": float(total),
              "per_stream_spread": spread_ratio, "flat_ok": bool(flat_ok), "stable_ok": True,
              "n_streams": len(PRIMARY_SUITE)}))
        valid_metric = (total > 0) and math.isfinite(total)
        sys.exit(0 if valid_metric else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
