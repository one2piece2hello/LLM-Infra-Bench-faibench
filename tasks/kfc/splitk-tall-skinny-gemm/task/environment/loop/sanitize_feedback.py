#!/usr/bin/env python3
"""Sanitized feedback emitter (0700 root-owned) for splitk-tall-skinny-gemm.

Invoked ONLY by /opt/loop/submit.sh. Reads the DEV engine products under
/logs/loop/dev/ and the loop accounting under /logs/loop/, and prints ONLY a
leak-free summary.

Emits (nothing else, ever):
  submission k/16
  correctness: PASS | FAIL
  [FAIL only] failing_invariant: <name from a fixed allow-list>
  [FAIL only] diagnostic: <short, actionable, leak-free>
  dev_speedup: <x>            (relative performance ratio vs the reference baseline; higher is better)
  dev_reward: <r>             (= dev_speedup when correct, else 0; an in-session PROXY)
  best_so_far: submission <j>, dev_speedup <x>, dev_reward <r>
  remaining: <16 - k>
  finalize_allowed: <true|false>
  [harness path] harness_error: <message>  (candidate not at fault; budget not consumed)

NEVER emits: verifier / test.sh / compute_reward contents; hidden workload
names / seeds / counts / shapes; any threshold; ref_speedup; absolute timings or
byte counts; actual_hardware_type.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LOOP = Path("/logs/loop")
DEV = LOOP / "dev"
MIN_SUBMISSIONS = 3
MAX_SUBMISSIONS = 16

SIGNATURE = 'gemm(A, B) -> C = A @ B, a (M, N) float32 CUDA tensor with the product accumulated in float32 and row order preserved; the multiply lives in gemm_kernel.cu behind the fixed gemm.py wrapper — do NOT change the public entry point'
PRIMITIVE_HINT = 'your implementation delegated the multiply to a prebuilt matrix-multiply library or framework primitive (cuBLAS / cublasLt / cutlass, libtorch at::matmul / torch::matmul / at::mm / torch::mm / .matmul(...), or torch.matmul / torch.mm / F.linear / the @ operator) — implement the multiply yourself in gemm_kernel.cu; if you split the inner dimension and sum partial results, add them in a fixed, well-defined order so the output is reproducible run to run; do not reference those tokens even in comments (the scan is textual)'


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def _named_invariant(state: dict):
    hard = [str(r) for r in (state.get("hard_fail_reasons") or [])]

    def has(*subs):
        return any(any(s in h for s in subs) for h in hard)

    if has("forbidden_edit_path", "forbidden_source"):
        return ("edit_scope",
                "only the single product file may change and it must not reference any "
                "scoring / verifier path — keep your edits to the product implementation only")
    if has("_primitive", "_operator"):
        return ("allowed_ops", PRIMITIVE_HINT)
    if has("product_file_missing", "candidate_load_failed", "repo_missing"):
        return ("product_present_and_importable",
                f"the product could not be found or imported — keep the public {SIGNATURE} intact")
    if has("hidden_correctness_failed"):
        return ("output_matches_reference",
                "your output disagreed with the reference on at least one case (numeric tolerance, "
                "dtype / shape / device, an error-contract probe that must raise, or a metamorphic "
                f"invariant) — verify the contract of {SIGNATURE}")
    if has("benchmark_execution_failed"):
        return ("runs_without_error",
                "the measurement could not complete — your implementation likely raised, produced "
                "non-finite values, or disagreed with the baseline while being measured")
    return ("verifier_completed",
            "the check did not complete cleanly — re-check that your last change did not cause an early crash")


def main() -> int:
    try:
        k = int((LOOP / "count").read_text(encoding="utf-8").strip())
    except Exception:
        k = 0

    he = DEV / "harness_error.txt"
    if he.exists():
        try:
            msg = he.read_text(encoding="utf-8").strip().splitlines()[0]
        except Exception:
            msg = "dev engine infrastructure error"
        print(f"submission {k}/{MAX_SUBMISSIONS}")
        print(f"harness_error: {msg} — this is NOT a defect in your code; do not debug the "
              f"harness. Retry or continue optimizing (this attempt was not counted).")
        return 0

    state = _read_json(DEV / "verifier_state.json", {})
    bench = _read_json(DEV / "benchmark_results.json", {})
    reward = _read_json(DEV / "reward.json", {})

    corr_ok = bool(state.get("correctness_ok"))
    speedup = bench.get("aggregate_speedup", None)
    r = reward.get("reward", 0.0)

    print(f"submission {k}/{MAX_SUBMISSIONS}")
    if corr_ok:
        print("correctness: PASS")
    else:
        print("correctness: FAIL")
        inv, hint = _named_invariant(state)
        print(f"failing_invariant: {inv}")
        print(f"diagnostic: {hint}")

    print(f"dev_speedup: {_fmt(speedup) if speedup is not None else 'n/a'}")
    print(f"dev_reward: {_fmt(r)}")

    best = _read_json(LOOP / "best.json", None)
    if best and isinstance(best, dict) and best.get("submission"):
        print(f"best_so_far: submission {best.get('submission')}, "
              f"dev_speedup {_fmt(best.get('speedup'))}, dev_reward {_fmt(best.get('reward'))}")
    else:
        print("best_so_far: (none — no correctness-passing submission yet)")

    print(f"remaining: {max(0, MAX_SUBMISSIONS - k)}")
    print(f"finalize_allowed: {'true' if k >= MIN_SUBMISSIONS else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
