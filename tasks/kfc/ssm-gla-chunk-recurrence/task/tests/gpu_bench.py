"""GPU benchmark for the gated running-state sequence mixer.

Per-shape paired timing of the frozen sequential baseline vs. the candidate;
per-shape ratio = baseline_median / candidate_median; final metric = geometric
mean across shapes (prints "speedup=X"). The value axis is latency: the naive
baseline walks the sequence one position at a time (a long dependency chain of
tiny launches), so a chunk-parallel reformulation that uses large matmuls wins on
long sequences. The baseline is loaded from the protected frozen copy
(KB_BASELINE_MODULE) so a no-op candidate (== frozen baseline) ties at ~1.0.
"""

import importlib.util
import os
import sys

import torch

from kb_gsr_harness import (
    BF16,
    forbidden_vendor_guard,
    geomean,
    ideal_mults,
    load_candidate,
    make_qkv,
)

# Long-sequence, head-dim-128 workloads where the sequential dependency chain
# dominates the naive baseline. (tag, B, H, L, D, seed)
SHAPES = [
    ("base",    8, 8, 2048, 128, 7000),
    ("long_t",  4, 8, 4096, 128, 7100),
    ("many_h",  8, 16, 1024, 128, 7200),
    ("short_t", 8, 8, 512, 128, 7300),
]

WARMUP = 2
ITERS = 3
RTOL = 5e-2
ATOL = 5e-2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/gated_state_recurrence.py")
    return _load(path).gated_state_recurrence


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


def _agree(a, b, tag):
    ca = a.to(torch.float32)
    cb = b.to(torch.float32)
    if ca.shape != cb.shape:
        print(f"BENCH_FAIL {tag}: o shape {tuple(ca.shape)} vs {tuple(cb.shape)}")
        return False
    if not torch.isfinite(ca).all():
        print(f"BENCH_FAIL {tag}: non-finite candidate o")
        return False
    diff = (ca - cb).abs()
    tol = ATOL + RTOL * cb.abs()
    if (diff > tol).any():
        print(f"BENCH_FAIL {tag}: {int((diff > tol).sum())} o elements disagree with baseline")
        return False
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = load_candidate().gated_state_recurrence
    baseline = _baseline_fn()

    ratios = []
    for tag, B, H, L, D, seed in SHAPES:
        q, k, v, g = make_qkv(B, H, L, D, D, seed, dtype=BF16)
        args = (q, k, v, g)

        with forbidden_vendor_guard():
            co = candidate(*args)
            torch.cuda.synchronize()
        bo = baseline(*args)
        torch.cuda.synchronize()
        co = co[0] if isinstance(co, (tuple, list)) else co
        bo = bo[0] if isinstance(bo, (tuple, list)) else bo
        if not _agree(co, bo, tag):
            print("speedup=0.0")
            sys.exit(1)
        del co, bo
        torch.cuda.empty_cache()

        base_ms = _time_median_ms(baseline, args, guard=False)
        cand_ms = _time_median_ms(candidate, args, guard=True)
        ratio = base_ms / max(cand_ms, 1e-6)
        ratios.append(ratio)
        print(f"shape={tag} B={B} H={H} L={L} D={D} baseline_ms={base_ms:.4f} "
              f"candidate_ms={cand_ms:.4f} ratio={ratio:.4f} ideal_mults={ideal_mults(B, H, L, D, D)}")
        del q, k, v, g
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
