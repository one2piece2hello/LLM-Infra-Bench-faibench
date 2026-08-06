"""GPU benchmark for the tall-skinny fp32 GEMM task.

Per-shape paired timing of the frozen baseline (compiled from the protected
/opt/verifier-baseline copy) vs. the candidate; per-shape ratio =
baseline_median / candidate_median; final metric = geometric mean across shapes
(prints "speedup=X"). Every benchmark shape is deliberately in the few-output /
large-inner-dimension regime (small M,N, large K) — that is where the frozen
single-block-per-tile baseline launches too few blocks to fill the device and a
solution that spreads the inner-dimension reduction across blocks wins. On
large square shapes there is no such headroom, so those are intentionally
excluded. A no-op candidate (== frozen baseline) ties at ~1.0. Also prints
achieved-GFLOP/s work evidence per shape.
"""

import sys

import torch

from kb_gemm_harness import (
    achieved_gflops,
    assert_close,
    geomean,
    load_baseline,
    load_candidate,
    make_ab,
    ref_gemm,
)

# Few-output / large-inner-dimension shapes (small M,N <= 256, large K >= 16384).
# This is the regime where partitioning the inner dimension across blocks pays
# off; square shapes are excluded on purpose (no occupancy headroom there).
SHAPES = [
    ("skinny_128_32k", 128, 128, 32768, 6000),
    ("skinny_256_32k", 256, 256, 32768, 6100),
    ("skinny_64_64k",   64,  64, 65536, 6200),
    ("rect_128x64_49k", 128,  64, 49152, 6300),
]

WARMUP = 2
ITERS = 5


def _time_median_ms(fn, A, B):
    for _ in range(WARMUP):
        fn(A, B)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(A, B)
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
    candidate = load_candidate().gemm
    baseline = load_baseline().gemm

    ratios = []
    for tag, M, N, K, seed in SHAPES:
        A, B = make_ab(M, N, K, seed)
        ref = ref_gemm(A, B)

        cy = candidate(A, B)
        torch.cuda.synchronize()
        try:
            assert_close(cy, ref, "C", f"[{tag}]")
        except AssertionError as exc:
            print(f"BENCH_FAIL {tag}: {exc}")
            print("speedup=0.0")
            sys.exit(1)
        del cy, ref
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, A, B)
        cand_ms = _time_median_ms(candidate, A, B)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        print(f"shape={tag} M={M} N={N} K={K} baseline_ms={base_ms:.4f} candidate_ms={cand_ms:.4f} "
              f"ratio={ratio:.4f} baseline_gflops={achieved_gflops(M, N, K, base_ms):.1f} "
              f"candidate_gflops={achieved_gflops(M, N, K, cand_ms):.1f}")
        del A, B
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
