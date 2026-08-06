#!/usr/bin/env python3
"""/opt/loop/sanitize_feedback.py — leak-free per-round feedback for
e2e-vllm-scheduler-mixed-batch-serving.

Invoked ONLY by /opt/loop/submit.sh. Reads the normalized DEV products under /logs/loop/dev/ and the
loop accounting under /logs/loop/, and prints ONLY a sanitized summary.

NEVER emits: the hidden burst workloads, the strong baseline, the calibrated ref_speedup, gate
thresholds, or any /tests content — NONE of which is in this container. Emits: submission index +
remaining, a server liveness/usability pass/fail (+ the NAMED failing invariant on fail), the RAW
PUBLIC-proxy serving throughput (explicitly NOT the graded score), best_so_far, and finalize_allowed.
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
    dev_score = reward.get("dev_score")
    dev_latency = reward.get("dev_latency_ms")
    remaining = max(0, MAX_SUB - count)
    finalize_allowed = count >= MIN_SUB

    print(f"submission {count}/{MAX_SUB}")
    if correctness_ok:
        print("server: OK  (launched and served the public dev prompts)")
        if isinstance(dev_score, (int, float)):
            lat = f"{float(dev_latency):.1f} ms" if isinstance(dev_latency, (int, float)) else "n/a"
            print(f"dev_throughput: {float(dev_score):.3f} (1000/median_ms; median latency {lat})   "
                  "(PUBLIC dev proxy — higher is better — NOT your graded score)")
    else:
        print("server: FAIL")
        print(f"  failing_invariant: {failing_inv or 'unknown'}")

    if best and best.get("submission"):
        bscore = best.get("dev_score")
        bs_s = f"{float(bscore):.3f}" if isinstance(bscore, (int, float)) else "n/a"
        print(f"best_so_far: submission {best.get('submission')}, dev_throughput {bs_s}")
    else:
        print("best_so_far: (no server-passing submission yet)")

    print(f"remaining: {remaining}")
    print(f"finalize_allowed: {'true' if finalize_allowed else 'false'} "
          f"(you may --finalize at any k>=1; there is no minimum and no 'k rounds without gain' rule)")
    print("NOTE: the dev proxy measures single-request latency + server liveness on public prompts. "
          "The GRADE also welds a greedy token-PARITY gate and times concurrent burst workloads — "
          "keep greedy outputs identical to the baseline; a fast-but-wrong server scores 0 at grading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
