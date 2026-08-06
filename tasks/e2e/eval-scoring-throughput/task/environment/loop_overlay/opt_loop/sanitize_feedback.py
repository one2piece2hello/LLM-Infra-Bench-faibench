#!/usr/bin/env python3
"""/opt/loop/sanitize_feedback.py — leak-free per-round feedback for
e2e-h3-eval-harness-throughput-quality.

Invoked ONLY by /opt/loop/submit.sh. Reads the normalized DEV products under /logs/loop/dev/ and the
loop accounting under /logs/loop/, and prints ONLY a sanitized summary.

NEVER emits: the held-out sample set, the strong-baseline reference, the calibrated ref_speedup /
strong_baseline_time_sec, hidden seeds/thresholds, or any /tests content — NONE of which is in this
container. Emits: submission index + remaining, correctness pass/fail (+ the NAMED failing gate on
fail), the RAW PUBLIC-proxy dev_speedup vs the naive template (explicitly NOT the graded score),
best_so_far, and finalize_allowed.
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
    dev_speedup = reward.get("dev_speedup")
    remaining = max(0, MAX_SUB - count)
    finalize_allowed = count >= MIN_SUB

    print(f"submission {count}/{MAX_SUB}")
    if correctness_ok:
        print("correctness: PASS  (per-sample consistency + anti-cache gates cleared on the dev split)")
        if isinstance(dev_speedup, (int, float)):
            print(f"dev_speedup: {float(dev_speedup):.4f}x   "
                  "(PUBLIC dev proxy vs the naive template — higher is better — NOT your graded score)")
        else:
            print("dev_speedup: n/a (timing not measured this round; correctness passed)")
    else:
        print("correctness: FAIL")
        print(f"  failing_invariant: {failing_inv or 'unknown'}")

    if best and best.get("submission"):
        bs = best.get("dev_speedup")
        bs_s = f"{float(bs):.4f}x" if isinstance(bs, (int, float)) else "n/a"
        print(f"best_so_far: submission {best.get('submission')}, dev_speedup {bs_s}")
    else:
        print("best_so_far: (no correctness-passing submission yet)")

    print(f"remaining: {remaining}")
    print(f"finalize_allowed: {'true' if finalize_allowed else 'false'} "
          f"(you may --finalize at any k>=1; there is no minimum and no 'k rounds without gain' rule)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
