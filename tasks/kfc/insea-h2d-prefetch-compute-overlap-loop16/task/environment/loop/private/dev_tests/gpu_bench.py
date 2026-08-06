"""GPU benchmark for the chunked host-to-device transfer / compute-overlap task.

Per-config paired timing of the frozen serial baseline vs. the candidate over the whole
streamed_chunk_apply call (host->device copies + per-chunk compute + concat); per-config
ratio = baseline_median / candidate_median; final metric = geometric mean across configs
(prints "speedup=X"). The value axis is single-H20 wall-clock latency: overlapping each
chunk's host->device copy with the previous chunk's compute drives total latency toward
max(copy, compute) instead of their sum. The baseline is loaded from the protected frozen
copy (KB_BASELINE_MODULE) so a no-op candidate (== frozen baseline) ties at ~1.0. Also
prints analytic host-to-device bytes moved per config as work evidence.
"""

import importlib.util
import os
import sys

import torch

from kb_overlap_harness import (
    BF16,
    forbidden_overlap_guard,
    geomean,
    make_chunks,
    make_compute,
    total_h2d_bytes,
)

# Single-H20 configs sized so per-chunk host->device copy time is comparable to per-chunk
# compute time (that is the regime where hiding the transfer behind compute pays off).
# (tag, n_chunks, rows_per_chunk, D, F, seed). Pinned shapes; re-tune if you change hardware.
CONFIGS = [
    ("balanced", 8, 4096, 4096, 4096, 6000),
    ("wide_xfer", 8, 2048, 8192, 4096, 6100),
    ("many_chunks", 16, 2048, 4096, 4096, 6200),
    ("tall", 6, 8192, 2048, 4096, 6300),
]

WARMUP = 2
ITERS = 6
RTOL = 5e-2
ATOL = 5e-2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate_fn():
    return _load(os.path.join(os.environ.get("KB_REPO_DIR", "/app/repo"), "streamed_apply.py")).streamed_chunk_apply


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/streamed_apply.py")
    return _load(path).streamed_chunk_apply


def _time_median_ms(fn, chunks, compute, guard=False):
    def call():
        if guard:
            with forbidden_overlap_guard():
                return fn(chunks, compute)
        return fn(chunks, compute)

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


def _agree(a, b, tag):
    ca = a.to(torch.float32)
    cb = b.to(torch.float32)
    if ca.shape != cb.shape:
        print(f"BENCH_FAIL {tag}: shape {tuple(ca.shape)} vs {tuple(cb.shape)}")
        return False
    if not torch.isfinite(ca).all():
        print(f"BENCH_FAIL {tag}: non-finite candidate")
        return False
    diff = (ca - cb).abs()
    tol = ATOL + RTOL * cb.abs()
    if (diff > tol).any():
        print(f"BENCH_FAIL {tag}: {int((diff > tol).sum())} elements disagree with baseline")
        return False
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = _candidate_fn()
    baseline = _baseline_fn()

    ratios = []
    for tag, n, rows, D, F, seed in CONFIGS:
        chunks = make_chunks([rows] * n, D, seed=seed, dtype=BF16)
        compute = make_compute(D, F, seed=seed + 1, dtype=BF16)

        with forbidden_overlap_guard():
            cout = candidate(chunks, compute)
            torch.cuda.synchronize()
        bout = baseline(chunks, compute)
        torch.cuda.synchronize()
        if not _agree(cout, bout, tag):
            print("speedup=0.0")
            sys.exit(1)
        del cout, bout
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, chunks, compute, guard=False)
        cand_ms = _time_median_ms(candidate, chunks, compute, guard=True)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        print(f"config={tag} n={n} rows={rows} D={D} F={F} baseline_ms={base_ms:.4f} "
              f"candidate_ms={cand_ms:.4f} ratio={ratio:.4f} h2d_bytes={total_h2d_bytes(chunks)}")
        del chunks, compute
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
