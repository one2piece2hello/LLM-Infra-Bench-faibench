"""GPU benchmark for the fp32 SGEMM task.

Per-shape paired timing of the frozen naive baseline (compiled from the protected
/opt/verifier-baseline copy) vs. the candidate; per-shape ratio =
baseline_median / candidate_median; final metric = geometric mean across shapes
(prints "speedup=X"). The op is compute-bound, so wall-time speedup tracks the
achieved fp32 throughput / arithmetic intensity. A no-op candidate (== frozen
baseline) ties at ~1.0. Also prints achieved-GFLOP/s work evidence per shape.
"""

import sys

import torch

from kb_sgemm_harness import (
    achieved_gflops,
    assert_close,
    geomean,
    load_baseline,
    load_candidate,
    make_abc,
    ref_sgemm,
)

# Compute-bound serving-shaped GEMMs: large square + one non-square, fp32.
# alpha/beta exercise both the product and the beta*C term.
SHAPES = [
    ("square_1k", 1024, 1024, 1024, 1.0, 0.0, 6000),
    ("square_2k", 2048, 2048, 2048, 1.0, 1.0, 6100),
    ("square_4k", 4096, 4096, 4096, 1.0, 0.0, 6200),
    ("rect_mnk",  2048, 4096, 1024, 0.75, 0.5, 6300),
]

WARMUP = 2
ITERS = 5


def _time_median_ms(fn, args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
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
    candidate = load_candidate().sgemm
    baseline = load_baseline().sgemm

    ratios = []
    for tag, M, N, K, alpha, beta, seed in SHAPES:
        A, B, C = make_abc(M, N, K, seed)
        ref = ref_sgemm(A, B, C, alpha, beta)
        args = (A, B, C, alpha, beta)

        cy = candidate(*args)
        torch.cuda.synchronize()
        try:
            assert_close(cy, ref, "D", f"[{tag}]")
        except AssertionError as exc:
            print(f"BENCH_FAIL {tag}: {exc}")
            print("speedup=0.0")
            sys.exit(1)
        del cy, ref
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, args)
        cand_ms = _time_median_ms(candidate, args)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        print(f"shape={tag} M={M} N={N} K={K} baseline_ms={base_ms:.4f} candidate_ms={cand_ms:.4f} "
              f"ratio={ratio:.4f} baseline_gflops={achieved_gflops(M, N, K, base_ms):.1f} "
              f"candidate_gflops={achieved_gflops(M, N, K, cand_ms):.1f}")
        del A, B, C
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
