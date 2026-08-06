"""Starter template for e2e-h3-eval-harness-throughput-quality (family C).

Copy this to /app/submission/scoring_pipeline.py and make it your own.

You are optimizing the END-TO-END THROUGHPUT of an LLM evaluation harness's SCORING/AGGREGATION
path — the lm-evaluation-harness filter-ensemble + metric stage (see lm_eval/api/task.py
`apply_filters`): regex answer extraction, take_first / majority_vote response transforms,
loglikelihood-choice accuracy, exact_match / contains / prefix_match. You are handed a FIXED set
of eval records that already contain the model's OUTPUTS (generated text or per-choice
loglikelihoods); your job is to turn them into per-sample scores AS FAST AS POSSIBLE.

Expose `load_scoring_pipeline_for_verification(device)` returning an object with:

    .score(samples: list[dict]) -> list[dict]     # each output row: {"id": <id>, "score": <float>}

The grader feeds ITS OWN held-out records, TIMES your `.score()` against a strong baseline it
RE-MEASURES in the same session (ABBA-alternating pairs, >= 5; speedup = median of
baseline_time / candidate_time), and RECOMPUTES the reference scores with an INDEPENDENT
implementation. Your reward is BOUNDED to [0.0, 1.0]:

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

where `ref_speedup` is a frozen constant calibrated at authoring time from a reference
solution. So: matching the strong baseline (speedup == 1.0) scores **0**, not 1.0; there is
NO credit below the baseline (speedup <= 1 -> 0); matching `ref_speedup` scores 0.5; and you
only reach the 1.0 cap at `ref_speedup` SQUARED. Speed only counts if every per-sample score
matches the reference EXACTLY. There is a HARD, WELDED
consistency gate: skip a sample, drop an id, or approximate a score and you get 0. An anti-cache
probe re-scores perturbed inputs under fresh ids, so a copied (id -> score) table from a dev run
also scores 0.

WHERE THE SPEED COMES FROM (all fair game): vectorise the regex extraction and metric compute
(batch over records, compile patterns once, group by metric/filter), avoid per-row Python overhead,
cache compiled regexes / tokenisations, exploit the auto-batch cache-clearing insight from
lm-evaluation-harness PR #3654, or rewrite lm_eval's scoring path in /app/repo. You may modify
anything in /app/repo. The DEV split under /data/eval_harness lets you check correctness + measure
your own speed before submitting; it is DISJOINT from the scored held-out set.

Record schema (same for dev and held-out):
  metric = "exact_match" | "contains" | "prefix_match" | "loglikelihood_acc"
  filter = "take_first" | "majority_vote"                          (generative metrics)
  filter_pattern = optional regex (group(1) if present else group(0) is the extracted answer)
  response = str OR list[str]                                       (generative)
  gold = str                                                        (generative)
  choice_loglikelihoods = list[float]; gold_index = int            (multiple-choice)

The reference definitions (which your scores MUST reproduce exactly):
  - normalise(text) = collapse whitespace, strip, lowercase.
  - filter: apply regex (group(1) else group(0), else ""); take_first -> first candidate;
    majority_vote -> most frequent normalised candidate, ties broken lexicographically smallest.
  - loglikelihood_acc: argmax over choice_loglikelihoods (ties -> smallest index) == gold_index.
  - exact_match: normalise(pred) == normalise(gold); contains: normalise(gold) in normalise(pred);
    prefix_match: normalise(pred).startswith(normalise(gold)).

THIS TEMPLATE is a deliberately naive, correct-but-slow per-row Python loop. It clears the
consistency gate but is SLOWER than the strong baseline, so as shipped it scores 0.0 — the
throughput headroom above the baseline is where the entire reward lives.
"""
from __future__ import annotations

import re


_WS = re.compile(r"\s+")


def _normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


class NaiveScoringPipeline:
    """Correct-but-slow reference-shaped scorer: one Python call per record, regex recompiled
    lazily. Intentionally leaves throughput on the table."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def _apply_filter(self, raw, filt, pattern) -> str:
        cands = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
        if pattern:
            rx = re.compile(pattern)  # recompiled every call on purpose (slow)
            out = []
            for c in cands:
                m = rx.search(c)
                out.append(m.group(1) if (m and m.groups()) else (m.group(0) if m else ""))
            cands = out
        if filt == "majority_vote":
            counts: dict[str, int] = {}
            for c in cands:
                key = _normalise(c)
                counts[key] = counts.get(key, 0) + 1
            best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            return best[0][0] if best else ""
        return cands[0] if cands else ""

    def _score_one(self, s: dict) -> float:
        metric = str(s.get("metric", "exact_match"))
        if metric == "loglikelihood_acc":
            lls = [float(x) for x in s.get("choice_loglikelihoods", [])]
            if not lls:
                return 0.0
            pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
            return 1.0 if pred == int(s.get("gold_index", -1)) else 0.0
        pred = self._apply_filter(s.get("response"), str(s.get("filter", "take_first")),
                                  s.get("filter_pattern"))
        gold = str(s.get("gold", ""))
        if metric == "exact_match":
            return 1.0 if _normalise(pred) == _normalise(gold) else 0.0
        if metric == "contains":
            return 1.0 if _normalise(gold) in _normalise(pred) else 0.0
        if metric == "prefix_match":
            return 1.0 if _normalise(pred).startswith(_normalise(gold)) else 0.0
        return 0.0

    def score(self, samples: list[dict]) -> list[dict]:
        return [{"id": s["id"], "score": self._score_one(s)} for s in samples]


def load_scoring_pipeline_for_verification(device: str = "cpu"):
    """REQUIRED entry point. Return an object exposing .score(samples) -> list[{'id','score'}]."""
    return NaiveScoringPipeline(device=device)
