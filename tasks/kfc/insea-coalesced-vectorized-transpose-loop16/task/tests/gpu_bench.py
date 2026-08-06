"""GPU benchmark for the 2-D transpose task.

Per-shape paired timing of the frozen baseline (compiled from the protected
/opt/verifier-baseline copy) vs. the candidate; per-shape ratio =
baseline_median / candidate_median; final metric = geometric mean across shapes
(prints "speedup=X"). The op is memory-bound, so wall-time speedup tracks the
achieved global-memory bandwidth. A no-op candidate (== frozen baseline) ties at
~1.0. Also prints achieved-GB/s work evidence per shape (bytes moved are the same
for a coalesced vs. a strided kernel — only the achieved bandwidth differs).
"""

import sys

import torch

from kb_transpose_harness import (
    FP16,
    FP32,
    achieved_gbps,
    assert_exact,
    geomean,
    load_baseline,
    load_candidate,
    make_matrix,
    ref_transpose,
)

# Memory-bound serving-shaped transposes: large square + skewed rectangles.
# (tag, M, N, dtype, itemsize, seed)
SHAPES = [
    ("square_8k",  8192,  8192, FP32, 4, 6000),
    ("rect_4x12k", 4096, 12288, FP32, 4, 6100),
    ("tall_16x2k", 16384, 2048, FP16, 2, 6200),
    ("wide_2x16k",  2048, 16384, FP16, 2, 6300),
]

WARMUP = 3
ITERS = 10


def _time_median_ms(fn, x):
    for _ in range(WARMUP):
        fn(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = load_candidate().transpose
    baseline = load_baseline().transpose

    ratios = []
    for tag, M, N, dtype, itemsize, seed in SHAPES:
        x = make_matrix(M, N, seed, dtype=dtype)
        ref = ref_transpose(x)

        cy = candidate(x)
        torch.cuda.synchronize()
        try:
            assert_exact(cy, ref, "y", f"[{tag}]")
        except AssertionError as exc:
            print(f"BENCH_FAIL {tag}: {exc}")
            print("speedup=0.0")
            sys.exit(1)
        del cy, ref
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, x)
        cand_ms = _time_median_ms(candidate, x)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        print(f"shape={tag} M={M} N={N} dtype={dtype} baseline_ms={base_ms:.4f} "
              f"candidate_ms={cand_ms:.4f} ratio={ratio:.4f} "
              f"baseline_gbps={achieved_gbps(M, N, itemsize, base_ms):.1f} "
              f"candidate_gbps={achieved_gbps(M, N, itemsize, cand_ms):.1f}")
        del x
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
