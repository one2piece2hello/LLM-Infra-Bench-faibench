"""e2e-h3-eval-harness-throughput-quality NEGATIVE controls (reviewer-only sanity checks).

Two negatives to prove BOTH gates during calibration:

(1) SlowButCorrectScorer — CORRECT scores (passes the consistency gate) but SLOW (naive per-row
    loop, regex recompiled every call). Expected reward < 1.0 (slower than the strong baseline):
    proves the reward is a genuine SPEED gradient, not a correctness break.

(2) SkippingScorer — FAST but SKIPS ~10% of samples (returns fewer ids). Expected reward = 0:
    proves the WELDED consistency gate (missing ids -> fail). Swap the loader at the bottom to
    activate whichever negative the calibration run needs.

Seeded by the negative VERIFIER_MODE at calibration time (seed dir: see tests/test.sh step 1).
"""
from __future__ import annotations

import re


_WS = re.compile(r"\s+")


def _normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def _score_one(s: dict) -> float:
    metric = str(s.get("metric", "exact_match"))
    if metric == "loglikelihood_acc":
        lls = [float(x) for x in s.get("choice_loglikelihoods", [])]
        if not lls:
            return 0.0
        pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
        return 1.0 if pred == int(s.get("gold_index", -1)) else 0.0
    raw = s.get("response")
    cands = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
    pattern = s.get("filter_pattern")
    if pattern:
        rx = re.compile(pattern)  # recompiled every call on purpose (slow)
        cands = [(lambda m: (m.group(1) if (m and m.groups()) else (m.group(0) if m else "")))(rx.search(c))
                 for c in cands]
    if str(s.get("filter", "take_first")) == "majority_vote":
        counts: dict[str, int] = {}
        for c in cands:
            counts[_normalise(c)] = counts.get(_normalise(c), 0) + 1
        pred = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else ""
    else:
        pred = cands[0] if cands else ""
    gold = str(s.get("gold", ""))
    if metric == "exact_match":
        return 1.0 if _normalise(pred) == _normalise(gold) else 0.0
    if metric == "contains":
        return 1.0 if _normalise(gold) in _normalise(pred) else 0.0
    if metric == "prefix_match":
        return 1.0 if _normalise(pred).startswith(_normalise(gold)) else 0.0
    return 0.0


class SlowButCorrectScorer:
    """Correct scores, slow implementation. Reward expected < 1.0 (not 0)."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def score(self, samples: list[dict]) -> list[dict]:
        return [{"id": s["id"], "score": _score_one(s)} for s in samples]


class SkippingScorer:
    """Fast but SKIPS every 10th sample. Reward expected == 0 (welded consistency gate)."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def score(self, samples: list[dict]) -> list[dict]:
        return [{"id": s["id"], "score": _score_one(s)} for i, s in enumerate(samples) if i % 10 != 0]


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    # Default negative = the SKIPPING scorer (proves the consistency gate -> 0).
    # For the slow-but-correct control, return SlowButCorrectScorer(device=device) instead.
    return SkippingScorer(device=device)
