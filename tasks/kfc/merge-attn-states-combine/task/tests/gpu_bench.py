"""GPU benchmark for the partial-attention-state combine task.

Per-shape paired timing of the frozen naive baseline vs. the candidate; per-shape
ratio = baseline_median / candidate_median; final metric = geometric mean across
shapes (prints "speedup=X"). The op is memory-bound, so the value axis is HBM bytes
moved and wall-time on H20 tracks it: the naive path materializes the full (N, R, D)
weighted-partial intermediate, while a single-pass combine never does. The baseline
is loaded from the protected frozen copy (KB_BASELINE_MODULE) so a no-op candidate
(== frozen baseline) ties at ~1.0. Also prints an analytic bytes-moved evidence line
per shape (single-pass ideal + the avoidable naive intermediate traffic).
"""

import importlib.util
import os
import sys

import torch

from kb_combine_harness import (
    F32,
    forbidden_vendor_guard,
    geomean,
    ideal_bytes_moved,
    load_candidate,
    make_partials,
    naive_extra_bytes,
)

# Serving-shaped split-KV combine: N = chunks, R = rows (tokens*heads), D = head dim.
# fp32 partials (as produced by the attention chunks). Memory-bound over (N, R, D).
SHAPES = [
    ("base",         8, 131072, 128, 7000),
    ("many_chunks", 16,  65536, 128, 7100),
    ("wide_d",       4, 131072, 256, 7200),
    ("small",        4,  32768, 128, 7300),
]

WARMUP = 3
ITERS = 10
RTOL = 3e-2
ATOL = 3e-2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/combine_attn_states.py")
    return _load(path).combine_attn_states


def _time_median_ms(fn, args, guard=False):
    def call():
        if guard:
            with forbidden_vendor_guard():
                return fn(*args)
        return fn(*args)

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _agree(a, b, tag, label):
    ca = a.to(torch.float32)
    cb = b.to(torch.float32)
    if ca.shape != cb.shape:
        print(f"BENCH_FAIL {tag}: {label} shape {tuple(ca.shape)} vs {tuple(cb.shape)}")
        return False
    # both may carry -inf (empty-row lse); require matching non-finite positions/values.
    a_nf = ~torch.isfinite(ca)
    b_nf = ~torch.isfinite(cb)
    if not torch.equal(a_nf, b_nf):
        print(f"BENCH_FAIL {tag}: non-finite {label} positions disagree with baseline")
        return False
    if b_nf.any() and not torch.equal(ca[b_nf], cb[b_nf]):
        print(f"BENCH_FAIL {tag}: non-finite {label} values disagree with baseline")
        return False
    fin = torch.isfinite(cb)
    diff = (ca[fin] - cb[fin]).abs()
    tol = ATOL + RTOL * cb[fin].abs()
    if (diff > tol).any():
        print(f"BENCH_FAIL {tag}: {int((diff > tol).sum())} {label} elements disagree with baseline")
        return False
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = load_candidate().combine_attn_states
    baseline = _baseline_fn()

    ratios = []
    for tag, N, R, D, seed in SHAPES:
        po, pl = make_partials(N, R, D, seed, dtype=F32)
        args = (po, pl)

        with forbidden_vendor_guard():
            co, cl = candidate(*args)
            torch.cuda.synchronize()
        bo, bl = baseline(*args)
        torch.cuda.synchronize()
        if not (_agree(co, bo, tag, "out") and _agree(cl, bl, tag, "lse")):
            print("speedup=0.0")
            sys.exit(1)
        del co, cl, bo, bl
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, args, guard=False)
        cand_ms = _time_median_ms(candidate, args, guard=True)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        ideal = ideal_bytes_moved(N, R, D, 4)   # fp32 = 4 bytes
        extra = naive_extra_bytes(N, R, D, 4)
        print(f"shape={tag} N={N} R={R} D={D} baseline_ms={base_ms:.4f} candidate_ms={cand_ms:.4f} "
              f"ratio={ratio:.4f} ideal_bytes_moved={ideal} naive_extra_bytes={extra}")
        del po, pl
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
