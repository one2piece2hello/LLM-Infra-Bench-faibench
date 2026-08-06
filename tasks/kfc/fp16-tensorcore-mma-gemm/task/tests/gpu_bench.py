"""GPU benchmark for the fp16 GEMM task.

Per-shape paired timing of the frozen baseline (compiled from the protected
/opt/verifier-baseline copy) vs. the candidate; per-shape ratio =
baseline_median / candidate_median; final metric = geometric mean across shapes
(prints "speedup=X"). The op is compute-bound, so wall-time speedup tracks the
achieved fp16 throughput. A no-op candidate (== frozen baseline) ties at ~1.0.
Also prints achieved-TFLOP/s work evidence per shape.
"""

import sys

import torch

from kb_gemm_harness import (
    achieved_tflops,
    assert_close,
    geomean,
    load_baseline,
    load_candidate,
    make_ab,
    ref_gemm,
)

# Compute-bound serving-shaped GEMMs: large square + rectangular, fp16.
SHAPES = [
    ("square_2k", 2048, 2048, 2048, 6000),
    ("square_4k", 4096, 4096, 4096, 6100),
    ("rect_mk",   1024, 4096, 4096, 6200),
    ("rect_nk",   4096, 1024, 4096, 6300),
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
              f"ratio={ratio:.4f} baseline_tflops={achieved_tflops(M, N, K, base_ms):.1f} "
              f"candidate_tflops={achieved_tflops(M, N, K, cand_ms):.1f}")
        del A, B
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
