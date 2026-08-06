"""Peak-GPU-memory measurement for the cross-entropy loss+gradient task.

VALUE AXIS = PEAK MEMORY, not wall-time. The naive baseline materialises several
full ``(N, V)`` buffers (log-probs, probabilities, gradient); a memory-frugal
implementation keeps only per-row running state and can overwrite the logits
buffer with the gradient in place, so its peak ``max_memory_allocated`` is far
lower at large ``V``. Latency may be ~1.0x -- that is EXPECTED and is NOT scored.

Per shape:
  1. correctness agreement: candidate output must match the frozen baseline
     within tolerance (on a fresh clone; the candidate may mutate in place);
  2. peak = ``torch.cuda.max_memory_allocated`` measured over a single call, with
     only the freshly generated inputs resident at reset (the input buffer is the
     shared floor for both); ratio = baseline_peak / candidate_peak.
Final metric = geometric mean of the per-shape peak-memory ratios. Prints
"speedup=X" (the peak-memory ratio; a no-op candidate == frozen baseline ties at
~1.0). The baseline is loaded from the protected frozen copy (KB_BASELINE_MODULE).
"""

import importlib.util
import os
import sys

import torch

from kb_ce_harness import (
    assert_grad_close,
    assert_loss_close,
    forbidden_ce_guard,
    geomean,
    load_candidate,
    make_labels,
    make_logits,
)

IGN = -100
LS = 0.0
IGNORE_FRAC = 0.1

# (tag, N, V, dtype). Vocab large enough that the baseline's (N,V) materialisation
# dominates peak memory (false-negative checklist item #3: a path-exercising workload).
SHAPES = [
    ("v32k_fp32", 8192, 32768, torch.float32),
    ("v128k_fp32", 8192, 128256, torch.float32),
    ("v64k_bf16", 8192, 65536, torch.bfloat16),
    ("v128k_bf16", 8192, 128256, torch.bfloat16),
]


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_ce_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_fn():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/cross_entropy_grad.py")
    return _load(path).cross_entropy_loss_grad


def _peak_of(fn, N, V, dtype, seed, guard):
    """max_memory_allocated over one call, inputs freshly generated (only they
    are resident at reset, so the input buffer is the shared floor)."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logits = make_logits(N, V, seed, dtype=dtype)
    labels = make_labels(N, V, seed + 1, ignore_frac=IGNORE_FRAC, ignore_index=IGN)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if guard:
        with forbidden_ce_guard():
            loss, grad = fn(logits, labels, IGN, LS)
    else:
        loss, grad = fn(logits, labels, IGN, LS)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del logits, labels, loss, grad
    torch.cuda.empty_cache()
    return int(peak)


def _agree(candidate, baseline, N, V, dtype, seed):
    """Candidate must match baseline within tolerance before its peak counts."""
    logits = make_logits(N, V, seed, dtype=dtype)
    labels = make_labels(N, V, seed + 1, ignore_frac=IGNORE_FRAC, ignore_index=IGN)
    b_loss, b_grad = baseline(logits.clone(), labels, IGN, LS)
    with forbidden_ce_guard():
        c_loss, c_grad = candidate(logits.clone(), labels, IGN, LS)
    bf16 = dtype == torch.bfloat16
    assert_loss_close(c_loss, b_loss, bf16=bf16, msg=f"[agree {N}x{V} {dtype}]")
    assert_grad_close(c_grad, b_grad, bf16=bf16, msg=f"[agree {N}x{V} {dtype}]")
    del logits, labels, b_loss, b_grad, c_loss, c_grad
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    candidate = load_candidate().cross_entropy_loss_grad
    baseline = _baseline_fn()

    ratios = []
    seed = 5000
    for tag, N, V, dtype in SHAPES:
        try:
            _agree(candidate, baseline, N, V, dtype, seed)
        except AssertionError as exc:
            print(f"BENCH_FAIL {tag}: {exc}")
            print("speedup=0.0")
            sys.exit(1)
        cand_peak = _peak_of(candidate, N, V, dtype, seed, guard=True)
        base_peak = _peak_of(baseline, N, V, dtype, seed, guard=False)
        ratio = base_peak / max(cand_peak, 1)
        ratios.append(ratio)
        print(f"shape={tag} N={N} V={V} dtype={dtype} "
              f"baseline_peak={base_peak} candidate_peak={cand_peak} ratio={ratio:.4f}")
        seed += 100

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
