"""Build the e2e-h3-eval-harness-throughput-quality scoring record set at Docker BUILD time.

Runs on a CPU-only build worker; NO internet needed (records are generated deterministically
from a fixed seed, so the held-out set is fully reproducible and self-contained). Writes:

  Agent-visible DEV split (under /data — the agent's progress monitor / correctness proxy):
    /data/eval_harness/dev_samples.jsonl   {"id","metric","filter","filter_pattern",
                                            "response"|"choice_loglikelihoods","gold"|"gold_index"}

  HELD-OUT split (Dockerfile moves this under /opt/verifier root-0700; the harness re-scores it
  with its OWN independent reference scorer and re-times it; the agent NEVER sees it):
    /tmp/heldout_samples.jsonl

Each record models one lm-evaluation-harness eval INSTANCE after generation, ready for the
filter-ensemble + metric path (see lm_eval/api/task.py apply_filters). The candidate's job is to
run that scoring path over the FULL set as FAST as possible while reproducing EVERY per-sample
score EXACTLY. DEV and HELD-OUT are same-schema, same-distribution, disjoint id spaces.

Record schema (matches compute_reward.py reference scorer):
  metric = "exact_match" | "contains" | "prefix_match" | "loglikelihood_acc"
  filter = "take_first" | "majority_vote"        (generative metrics only)
  filter_pattern = optional regex; group(1) if present else group(0) is the extracted answer
  response = str OR list[str] (multiple sampled completions -> majority_vote)   (generative)
  gold = str                                                                    (generative)
  choice_loglikelihoods = list[float]; gold_index = int                         (multiple-choice)
"""
from __future__ import annotations

import json
import random
import string
from pathlib import Path

DATA_DIR = Path("/data/eval_harness")
TMP_DIR = Path("/tmp")

SEED = 20240725
DEV_SAMPLES = 4000
# 20000 rows x ~2 KB of response text = ~40 MB and a ~160 ms timed pass for the strong baseline.
# Two independent reasons for that size: the median-of-5 timing is stable to ~1.7% at this scale
# (an earlier 80-byte-per-row revision swung 16%, which was a third of the entire ceiling gap and
# would have made the reward mostly noise), and 5 x 160 ms leaves a wide margin under
# max_score_time_sec even for a candidate several times slower than the naive template.
HELDOUT_SAMPLES = 20000

_ANSWER_RX = r"answer\s*is\s*([A-D])"     # a common lm-eval extraction pattern
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
          "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def _rand_text(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _cot_words(rng: random.Random) -> int:
    """Response length in words, HEAVY-TAILED like a real chain-of-thought eval run.

    This distribution is load-bearing for the whole task, so it is worth being explicit about why.
    An earlier revision emitted ~10-word stub responses (~80 bytes). Measured on that shape, the
    per-row scoring cost was ~1us of interpreter overhead spread thinly across dict lookups and
    function calls, with no hot primitive to attack: a hand-written str.find matcher that provably
    matched `re` exactly came out SLOWER than the regex (1.01x), and a pyarrow/RE2 columnar rewrite
    was slower still (0.79x) because marshalling 80-byte Python strings into Arrow costs more than
    the C++ kernels save. The best achievable speedup was ~1.4x and three independent attempts to
    beat it all failed -- i.e. the task had no real headroom.

    Real harness scoring does not run on 80-byte responses; it runs on model reasoning traces. At
    these lengths the cost becomes a LINEAR SCAN over thousands of characters, which is attackable
    from several independent directions (memchr-style literal prefiltering instead of the regex
    engine, C-level split/join instead of a `\\s+` substitution, early exit once enough of a prefix
    is known, columnar batching that amortises per-row dispatch). Measured on this shape the same
    stdlib ceiling reaches 2.7x, and the contract floor is ~74x away, so the work being removed is
    real work rather than measurement overhead.

    Shape: 55% terse, 30% a few reasoning steps, 15% long deliberation. Mean ~2 KB per row.
    """
    r = rng.random()
    if r < 0.55:
        return rng.randint(8, 30)
    if r < 0.85:
        return rng.randint(60, 200)
    return rng.randint(300, 900)


def _make_record(prefix: str, i: int, rng: random.Random) -> dict:
    kind = rng.random()
    rid = f"{prefix}_s{i}"
    if kind < 0.4:
        # multiple-choice loglikelihood record
        n_choices = rng.choice([2, 4, 4, 5])
        lls = [round(rng.uniform(-30.0, -1.0), 6) for _ in range(n_choices)]
        gold = rng.randrange(n_choices)
        return {"id": rid, "metric": "loglikelihood_acc",
                "choice_loglikelihoods": lls, "gold_index": gold}
    if kind < 0.7:
        # generative exact_match with a regex "answer is X" extraction + take_first
        letter = rng.choice(list("ABCD"))
        noise = _rand_text(rng, _cot_words(rng))
        resp = f"{noise}. Therefore the answer is {letter}."
        gold = letter if rng.random() < 0.6 else rng.choice(list("ABCD"))
        return {"id": rid, "metric": "exact_match", "filter": "take_first",
                "filter_pattern": _ANSWER_RX, "response": resp, "gold": gold}
    if kind < 0.88:
        # generative majority_vote over several sampled completions (self-consistency)
        letter = rng.choice(list("ABCD"))
        k = rng.choice([3, 5, 5, 7])
        comps = []
        for _ in range(k):
            l = letter if rng.random() < 0.65 else rng.choice(list("ABCD"))
            comps.append(f"{_rand_text(rng, _cot_words(rng))} the answer is {l}")
        gold = letter if rng.random() < 0.6 else rng.choice(list("ABCD"))
        return {"id": rid, "metric": "exact_match", "filter": "majority_vote",
                "filter_pattern": _ANSWER_RX, "response": comps, "gold": gold}
    # free-form contains / prefix_match (no regex)
    gold = _rand_text(rng, rng.randint(1, 3))
    if rng.random() < 0.5:
        resp = (f"{_rand_text(rng, _cot_words(rng))} {gold} "
                f"{_rand_text(rng, _cot_words(rng))}").strip()
        metric = "contains"
    else:
        resp = f"{gold} {_rand_text(rng, _cot_words(rng))}".strip() if rng.random() < 0.7 \
            else _rand_text(rng, _cot_words(rng))
        metric = "prefix_match"
    return {"id": rid, "metric": metric, "filter": "take_first",
            "filter_pattern": None, "response": resp, "gold": gold}


def _build_split(prefix: str, n: int, rng: random.Random) -> list[dict]:
    return [_make_record(prefix, i, rng) for i in range(n)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dev = _build_split("dev", DEV_SAMPLES, random.Random(SEED + 1))
    heldout = _build_split("test", HELDOUT_SAMPLES, random.Random(SEED + 2))
    _write_jsonl(DATA_DIR / "dev_samples.jsonl", dev)
    _write_jsonl(TMP_DIR / "heldout_samples.jsonl", heldout)
    print(f"dev: {len(dev)} records; test(held-out): {len(heldout)} records", flush=True)


if __name__ == "__main__":
    main()
