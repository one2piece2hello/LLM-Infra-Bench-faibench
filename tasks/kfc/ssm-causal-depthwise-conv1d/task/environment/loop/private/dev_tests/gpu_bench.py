"""GPU benchmark for the per-channel trailing-window weighted-sum task.

Per-shape paired timing of the frozen naive baseline vs. the candidate; per-shape
ratio = baseline_median / candidate_median; final metric = geometric mean across
shapes (prints "speedup=X"). The op is launch/bandwidth-bound (padded copy + K
per-tap HBM passes + a separate activation pass in the naive form), so wall-time
speedup tracks the fused-single-pass value axis. The baseline is loaded from the
protected frozen copy (KB_BASELINE_MODULE) so a no-op candidate (== frozen
baseline) ties at ~1.0. Also prints an analytic bytes-moved evidence line.
"""

import importlib.util
import os
import sys

import torch

from kb_conv_harness import (
    BF16,
    forbidden_vendor_guard,
    geomean,
    ideal_bytes_moved,
    load_candidate,
    make_bias,
    make_w,
    make_x,
)

# (tag, B, C, L, seed). K fixed small (typical short-filter length). bf16.
# Launch/bandwidth-bound over (B, C, L); long L / many rows expose the headroom.
K = 4
SHAPES = [
    ("base", 4, 1024, 4096, 6000),
    ("wide_c", 2, 4096, 2048, 6100),
    ("long_l", 2, 1024, 8192, 6200),
    ("small", 8, 512, 2048, 6300),
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
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/channel_window_op.py")
    return _load(path).channel_window_op


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
    candidate = load_candidate().channel_window_op
    baseline = _baseline_fn()

    ratios = []
    for tag, B, C, L, seed in SHAPES:
        x = make_x(B, C, L, seed, dtype=BF16)
        w = make_w(C, K, seed + 1, dtype=BF16)
        bias = make_bias(C, seed + 2, dtype=BF16)
        args = (x, w, bias)

        with forbidden_vendor_guard():
            cy = candidate(*args)
            torch.cuda.synchronize()
        by = baseline(*args)
        torch.cuda.synchronize()
        if not _agree(cy, by, tag, "y"):
            print("speedup=0.0")
            sys.exit(1)
        del cy, by
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, args, guard=False)
        cand_ms = _time_median_ms(candidate, args, guard=True)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        ib = ideal_bytes_moved(B, C, L, K, 2)  # bf16 = 2 bytes
        print(f"shape={tag} B={B} C={C} L={L} K={K} baseline_ms={base_ms:.4f} "
              f"candidate_ms={cand_ms:.4f} ratio={ratio:.4f} ideal_bytes_moved={ib}")
        del x, w, bias
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
