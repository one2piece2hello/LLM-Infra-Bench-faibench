"""GPU benchmark for the frozen-linear + low-rank-correction apply task.

Per-shape paired timing of the frozen naive baseline (materialized [N, K] delta)
vs. the candidate; per-shape ratio = baseline_median / candidate_median; final
metric = geometric mean across shapes (prints "speedup=X"). The value axis is
FLOPs + peak memory: the efficient two-matmul path keeps the correction at
O(M*r*(K+N)) FLOPs and never allocates the [N, K] delta, so both compute and the
delta buffer shrink. The baseline is loaded from the protected frozen copy
(KB_BASELINE_MODULE) so a no-op candidate (== frozen baseline) ties at ~1.0.
Also prints analytic FLOPs / peak-delta-bytes work evidence per shape.
"""

import importlib.util
import os
import sys

import torch

from kb_lowrank_harness import (
    BF16,
    base_gemm_flops,
    correction_flops_lowrank,
    correction_flops_materialized,
    delta_peak_bytes,
    geomean,
    load_candidate,
    make_base_weight,
    make_factors,
    make_tensor,
)

# Training/serving-shaped: M = token rows (B*S), K = in features, N = out features,
# r = small rank. Chosen so forming/reading the [N, K] delta is a real fraction of
# the base GEMM (modest M, large N,K, tiny r) -> the low-rank path's saving shows.
SHAPES = [
    ("base",  2048, 4096, 4096,  8,  6000),
    ("wide",  1024, 8192, 8192,  16, 6100),
    ("mlp",   4096, 4096, 11008, 8,  6200),
    ("small", 512,  4096, 4096,  8,  6300),
]

WARMUP = 3
ITERS = 10
# candidate-vs-baseline sanity agreement (both bf16). The naive baseline rounds the
# merged [N,K] weight to bf16 while the two-matmul candidate does not, so the two
# correct bf16 paths can differ by up to ~5e-2 on large K; matched to the correctness
# harness bf16 tolerance (oracle-mode recalibration).
RTOL = 5e-2
ATOL = 5e-2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/lowrank_adapter_apply.py")
    return _load(path).lowrank_adapter_apply


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


def _agree(a, b, tag, label):
    ca = a.to(torch.float32)
    cb = b.to(torch.float32)
    if ca.shape != cb.shape:
        print(f"BENCH_FAIL {tag}: {label} shape {tuple(ca.shape)} vs {tuple(cb.shape)}")
        return False
    if not torch.isfinite(ca).all():
        print(f"BENCH_FAIL {tag}: non-finite candidate {label}")
        return False
    diff = (ca - cb).abs()
    tol = ATOL + RTOL * cb.abs()
    if (diff > tol).any():
        print(f"BENCH_FAIL {tag}: {int((diff > tol).sum())} {label} elements disagree with baseline")
        return False
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = load_candidate().lowrank_adapter_apply
    baseline = _baseline_fn()

    ratios = []
    for tag, M, K, N, r, seed in SHAPES:
        x = make_tensor((M, K), seed, dtype=BF16)
        W = make_base_weight(N, K, seed + 10, dtype=BF16)
        A, B = make_factors(N, K, r, seed + 20, dtype=BF16)
        scale = 2.0
        args = (x, W, A, B, scale)

        cy = candidate(*args)
        torch.cuda.synchronize()
        by = baseline(*args)
        torch.cuda.synchronize()
        if not _agree(cy, by, tag, "y"):
            print("speedup=0.0")
            sys.exit(1)
        del cy, by
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, args)
        cand_ms = _time_median_ms(candidate, args)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        corr_lr = correction_flops_lowrank(M, K, N, r)
        corr_mat = correction_flops_materialized(M, K, N, r)
        dbytes = delta_peak_bytes(N, K, 2)  # bf16 = 2 bytes
        base_f = base_gemm_flops(M, K, N)
        print(f"shape={tag} M={M} K={K} N={N} r={r} baseline_ms={base_ms:.4f} candidate_ms={cand_ms:.4f} "
              f"ratio={ratio:.4f} base_gemm_flops={base_f} corr_flops_lowrank={corr_lr} "
              f"delta_form_flops={corr_mat} delta_peak_bytes={dbytes}")
        del x, W, A, B
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
