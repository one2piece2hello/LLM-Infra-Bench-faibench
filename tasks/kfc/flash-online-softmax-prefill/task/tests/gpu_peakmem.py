"""GPU peak-memory benchmark for the causal-attention task.

Per-shape peak-memory measurement of the frozen naive baseline vs. the candidate;
per-shape ratio = baseline_peak_bytes / candidate_peak_bytes; final metric =
geometric mean across shapes (prints "speedup=X"). The naive baseline materializes
the (S,S) similarity + probability matrices (peak ~ O(B*H*S*S)); a candidate that
never forms that matrix keeps a ~O(B*H*S*D) working set, so max_memory_allocated
is the value axis. The baseline is loaded from the protected frozen copy
(KB_BASELINE_MODULE) so a no-op candidate (== frozen baseline) ties at ~1.0.
Also prints an analytic work-evidence line per shape.
"""

import importlib.util
import os
import sys

import torch

from kb_attn_harness import (
    BF16,
    forbidden_vendor_guard,
    geomean,
    make_qkv,
    naive_score_elems,
    working_set_elems,
)

# (tag, B, H, Hk, S, D, seed) -- causal, bf16. Peak memory dominated by the S*S term.
SHAPES = [
    ("base", 2, 16, 4, 4096, 128, 6000),
    ("wide_s", 1, 16, 4, 8192, 64, 6100),
    ("mid", 2, 16, 8, 2048, 128, 6200),
    ("small", 4, 8, 2, 2048, 64, 6300),
]

RTOL = 3e-2
ATOL = 3e-2
MEM_REPEATS = 2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/causal_attention.py")
    return _load(path).causal_attention


def _peak_extra_bytes(fn, args, guard=False):
    """Peak GPU bytes allocated *by the call* (high-water minus the pre-call
    live set). Deterministic across runs; take the max over a couple of repeats."""
    def call():
        if guard:
            with forbidden_vendor_guard():
                return fn(*args)
        return fn(*args)

    peaks = []
    for _ in range(MEM_REPEATS):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = call()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        peaks.append(max(peak - before, 1))
        del out
        torch.cuda.empty_cache()
    return max(peaks)


def _agree(a, b, tag):
    ca = a.to(torch.float32)
    cb = b.to(torch.float32)
    if ca.shape != cb.shape:
        print(f"BENCH_FAIL {tag}: shape {tuple(ca.shape)} vs {tuple(cb.shape)}")
        return False
    if not torch.isfinite(ca).all():
        print(f"BENCH_FAIL {tag}: non-finite candidate output")
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
    candidate = _load(os.environ.get("KB_CANDIDATE_MODULE", "/app/repo/causal_attention.py")).causal_attention
    baseline = _baseline_fn()

    ratios = []
    for tag, B, H, Hk, S, D, seed in SHAPES:
        q, k, v = make_qkv(B, H, Hk, S, D, seed, dtype=BF16)
        scale = 1.0 / (D ** 0.5)
        args = (q, k, v, scale, True)

        with forbidden_vendor_guard():
            cout = candidate(*args)
            torch.cuda.synchronize()
        bout = baseline(*args)
        torch.cuda.synchronize()
        if not _agree(cout, bout, tag):
            print("speedup=0.0")
            sys.exit(1)
        del cout, bout
        torch.cuda.empty_cache()

        base_bytes = _peak_extra_bytes(baseline, args, guard=False)
        cand_bytes = _peak_extra_bytes(candidate, args, guard=True)
        ratio = base_bytes / max(cand_bytes, 1)
        ratios.append(ratio)
        naive_sq = naive_score_elems(B, H, S)      # elements in the (S,S) term
        work = working_set_elems(B, H, S, D)        # elements in a linear working set
        print(f"shape={tag} B={B} H={H} Hk={Hk} S={S} D={D} "
              f"baseline_peak_bytes={base_bytes} candidate_peak_bytes={cand_bytes} "
              f"ratio={ratio:.4f} naive_score_elems={naive_sq} working_set_elems={work}")
        del q, k, v
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
