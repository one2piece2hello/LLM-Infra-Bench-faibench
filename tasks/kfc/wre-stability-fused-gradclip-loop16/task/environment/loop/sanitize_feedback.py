#!/usr/bin/env python3
# ============================================================================
# wro loop16 finalizer — sanitize_feedback.py  (REUSABLE, task-agnostic)  0700 root
# ============================================================================
"""Leak-free per-round feedback. Invoked ONLY by /opt/loop/submit.sh. Reads the
normalized DEV products under /logs/loop/dev/ and the loop accounting under
/logs/loop/, and prints ONLY a sanitized summary:

  submission k/16
  correctness: PASS | FAIL
  [FAIL only]  failing_invariant: <named, leak-free>
  dev_speedup: <x>
  dev_reward: <r>
  best_so_far: submission <j>, dev_speedup <x>, dev_reward <r>
  remaining: <16 - k>
  finalize_allowed: <true|false>
  [harness path] harness_error: <message>  (candidate not at fault)

NEVER emits: verifier/test.sh/compute_reward contents; hidden workload
names/seeds/counts/shapes; any threshold; ref_speedup; absolute timings.
MAY emit: pass/fail; a named invariant; relative dev_speedup; dev_reward;
submission index; best-so-far; remaining budget; finalize_allowed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

LOOP = Path("/logs/loop")
DEV = LOOP / "dev"
MIN_SUBMISSIONS = 3
MAX_SUBMISSIONS = 16

# Map a raw hard-fail reason (leak-free by construction) to a friendly, actionable
# invariant name. Reasons the wro verifier emits: out_of_scope_edit:<f>,
# import_origin_not_app_repo, correctness_trace_mismatch, timing_invalid,
# candidate_timing_failed, baseline_timing_failed, repo_missing,
# hidden_supplement_missing:<f>, oracle_apply_failed/baseline2_apply_failed.
_HINTS = {
    "edit_scope": "your diff changed a file outside the declared editable scope — "
                  "keep edits to the in-scope product file(s) only",
    "import_origin": "the module under test did not resolve to the in-repo tree — "
                     "do not shadow it with a copy elsewhere",
    "correctness": "your output no longer matches the reference behavior exactly — "
                   "the change must be speed-only, behavior-identical",
    "benchmark": "the timing run did not produce a valid measurement — check your "
                 "change does not crash or hang the benchmarked path",
    "verifier_completed": "the check did not complete — re-check your last edit "
                          "did not cause an early crash",
}


def _named_invariant(hard: list[str]) -> str:
    h0 = (hard[0] if hard else "")
    def startswith(*p): return any(any(x.startswith(pp) for pp in p) for x in hard)
    if startswith("out_of_scope_edit", "forbidden_edit_path", "forbidden_source_reference"):
        return f"edit_scope: {_HINTS['edit_scope']}"
    if startswith("import_origin"):
        return f"import_origin: {_HINTS['import_origin']}"
    if startswith("correctness_trace_mismatch", "correctness_failed", "correctness"):
        return f"output_matches_reference: {_HINTS['correctness']}"
    if startswith("timing_invalid", "candidate_timing_failed", "baseline_timing_failed", "benchmark"):
        return f"runs_without_error: {_HINTS['benchmark']}"
    if h0:
        return f"verifier_completed: {_HINTS['verifier_completed']}"
    return f"verifier_completed: {_HINTS['verifier_completed']}"


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
        print(f"harness_error: {msg} — this is NOT a defect in your code; do not "
              f"debug the harness. Retry or continue optimizing (this attempt was "
              f"not counted).")
        return 0

    state = _read_json(DEV / "verifier_state.json", {})
    reward = _read_json(DEV / "reward.json", {})
    corr_ok = bool(state.get("correctness_ok"))
    hard = [str(r) for r in (state.get("hard_fail_reasons") or [])]
    speedup = reward.get("speedup", None)
    r = reward.get("reward", 0.0)

    print(f"submission {k}/{MAX_SUBMISSIONS}")
    if corr_ok:
        print("correctness: PASS")
    else:
        print("correctness: FAIL")
        print(f"failing_invariant: {_named_invariant(hard)}")

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
