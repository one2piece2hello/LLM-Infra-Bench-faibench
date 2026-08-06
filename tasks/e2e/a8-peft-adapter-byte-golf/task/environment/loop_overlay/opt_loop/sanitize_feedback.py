#!/usr/bin/env python3
"""/opt/loop/sanitize_feedback.py — emit ONLY the leak-free per-round feedback for
e2e-a8-peft-adapter-byte-golf.

Reads /logs/loop/dev/{verifier_state.json,reward.json,harness_error.txt} and
/logs/loop/{count,best.json}; prints the sanitized block the solver reads.

NEVER emits: the held-out corpus, the calibrated strong_baseline_ce / ref_speedup, the hidden
seeds / token budget / thresholds, or any /tests content — NONE of which is even in this
container. Emits: submission index + remaining, correctness pass/fail (+ the NAMED failing
invariant on fail), the RAW public-proxy adaptation gain (explicitly NOT the graded score),
best_so_far, and finalize_allowed. The dev signal is a public proxy on a small public token
budget; the graded score is computed after the session over the hidden test split.
"""
from __future__ import annotations

import json
from pathlib import Path

LOOP = Path("/logs/loop")
DEV = LOOP / "dev"
MIN_SUB = 1
MAX_SUB = 16


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> int:
    count = 0
    cf = LOOP / "count"
    if cf.is_file():
        try:
            count = int("".join(ch for ch in cf.read_text() if ch.isdigit()) or "0")
        except Exception:
            count = 0

    # harness_error path: submit.sh has already refunded; say so and stop.
    he = DEV / "harness_error.txt"
    if he.is_file() and he.stat().st_size > 0:
        print(f"submission {count}/{MAX_SUB} — harness_error")
        print("harness_error: the DEV scoring engine itself failed — this is NOT a defect in your")
        print("code and this attempt was REFUNDED (your budget is unchanged). Do NOT debug the")
        print("harness; just retry submit, or keep optimizing and retry.")
        return 0

    state = _load(DEV / "verifier_state.json")
    reward = _load(DEV / "reward.json")
    best = _load(LOOP / "best.json")

    correctness_ok = bool(state.get("correctness_ok"))
    failing_inv = state.get("failing_invariant") or ""
    dev_gain = reward.get("dev_gain")
    remaining = max(0, MAX_SUB - count)
    finalize_allowed = count >= MIN_SUB

    print(f"submission {count}/{MAX_SUB}")
    if correctness_ok:
        print("correctness: PASS")
        if isinstance(dev_gain, (int, float)):
            print(f"dev_gain: {float(dev_gain):.4f}   "
                  "(PUBLIC dev proxy: base_ce - candidate_ce — higher is better — NOT your graded score)")
    else:
        print("correctness: FAIL")
        print(f"  failing_invariant: {failing_inv or 'unknown'}")
    # uid-drop transparency: the per-round training runs unprivileged, same as the grade.
    if state.get("dev_uid_dropped") is not None:
        print(f"dev_training_uid_dropped: {'true' if state.get('dev_uid_dropped') else 'false'}"
              f" (observed euid {state.get('observed_euid')})")

    if best and best.get("submission"):
        bb = best.get("dev_gain")
        bb_s = f"{float(bb):.4f}" if isinstance(bb, (int, float)) else "n/a"
        print(f"best_so_far: submission {best.get('submission')}, dev_gain {bb_s}")
    else:
        print("best_so_far: (no correctness-passing submission yet)")

    print(f"remaining: {remaining}")
    print(f"finalize_allowed: {'true' if finalize_allowed else 'false'} "
          f"(you may --finalize at any k>=1; there is no minimum and no 'k rounds without gain' rule)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
