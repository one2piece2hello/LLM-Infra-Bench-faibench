"""GPU benchmark for the first-order state-space scan task.

Per-shape paired timing of the frozen naive sequential baseline vs. the candidate;
per-shape ratio = baseline_median / candidate_median; final metric = geometric mean
across shapes (prints "speedup=X"). The value axis is latency via parallel depth: the
sequential loop serializes O(L) dependent steps, so a scan that parallelizes the time
axis on the GPU wins on wall-time for long L. The baseline is loaded from the
protected frozen copy (KB_BASELINE_MODULE) so a no-op candidate (== frozen baseline)
ties at ~1.0. Also prints an analytic sequential-depth work-evidence line per shape.
"""

import importlib.util
import os
import sys

import torch

from kb_scan_harness import (
    F32,
    analytic_proxy,
    forbidden_scan_guard,
    geomean,
    load_candidate,
    make_inputs,
)

# Long time axis is where the parallel-depth advantage shows. f32 primary dtype.
# (name, B, L, D, N, seed)
SHAPES = [
    ("base",       4, 1024, 512, 16, 6000),
    ("long_L",     2, 4096, 256, 16, 6100),
    ("wide_state", 4, 1024, 256, 64, 6200),
    ("many_batch", 16, 512, 256, 16, 6300),
]

WARMUP = 3
ITERS = 10
RTOL = 1e-2
ATOL = 1e-2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/state_space_scan.py")
    return _load(path).state_space_scan


def _time_median_ms(fn, args, guard=False):
    def call():
        if guard:
            with forbidden_scan_guard():
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


def _agree(a, b, tag):
    ca = a.to(torch.float32)
    cb = b.to(torch.float32)
    if ca.shape != cb.shape:
        print(f"BENCH_FAIL {tag}: y shape {tuple(ca.shape)} vs baseline {tuple(cb.shape)}")
        return False
    if not torch.isfinite(ca).all():
        print(f"BENCH_FAIL {tag}: non-finite candidate y")
        return False
    diff = (ca - cb).abs()
    tol = ATOL + RTOL * cb.abs()
    if (diff > tol).any():
        print(f"BENCH_FAIL {tag}: {int((diff > tol).sum())} y elements disagree with baseline")
        return False
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = load_candidate().state_space_scan
    baseline = _baseline_fn()

    ratios = []
    for tag, Bsz, L, D, N, seed in SHAPES:
        A, B, C, x = make_inputs(Bsz, L, D, N, seed, dtype=F32)
        args = (A, B, C, x)

        with forbidden_scan_guard():
            cy = candidate(*args)
            torch.cuda.synchronize()
        by = baseline(*args)
        torch.cuda.synchronize()
        if not _agree(cy, by, tag):
            print("speedup=0.0")
            sys.exit(1)
        del cy, by
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, args, guard=False)
        cand_ms = _time_median_ms(candidate, args, guard=True)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        pr = analytic_proxy(Bsz, L, D, N)
        print(f"shape={tag} B={Bsz} L={L} D={D} N={N} baseline_ms={base_ms:.4f} "
              f"candidate_ms={cand_ms:.4f} ratio={ratio:.4f} "
              f"seq_depth={pr['seq_depth']} log_depth={pr['log_depth']} state_elems={pr['state_elems']}")
        del A, B, C, x
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
