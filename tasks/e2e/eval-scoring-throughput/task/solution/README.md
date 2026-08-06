# e2e-h3-eval-harness-throughput-quality — ORACLE / strong-baseline (reviewer-only)

**NOT model-visible. Never COPYed into the image.** Seeded fresh at scoring via the
`strong_baseline` VERIFIER_MODE from `/opt/strong_baseline` at calibration time.

## What the task measures
Family C (quality-under-budget), single-shot, eval-only, **deterministic**. The candidate submits
`scoring_pipeline.py` exposing `load_scoring_pipeline_for_verification(device)` -> object with
`.score(samples) -> list[{"id","score"}]`. This is an LLM **evaluation-harness** optimization: the
candidate must run the lm-evaluation-harness filter-ensemble + metric path (regex answer
extraction, take_first / majority_vote transforms, loglikelihood-choice accuracy, exact_match /
contains / prefix_match) over a FIXED held-out record set **as fast as possible**, while producing
**exactly** the reference per-sample scores.

- **Metric / reward** (the bench reward spec 性能类, BOUNDED [0,1] since 2026-07-27):
  `reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0` with `ref_speedup = 2.24419` (frozen
  authoring-time constant; the oracle is never run at scoring) and
  `speedup = median` over `>=5` alternating ABBA pairs of `baseline_time / candidate_time`, the
  strong baseline RE-MEASURED in-session. Merely MATCHING the strong baseline is `speedup == 1.0`
  and scores **0**, not 1.0; `speedup == ref_speedup` scores 0.5; the 1.0 cap needs
  `speedup >= ref_speedup^2 = 5.0364`.
  [SUPERSEDED: the pre-migration form was the unbounded `strong_baseline_time / candidate_time`.]
  `0` on any hard failure: consistency gate (any per-sample
  mismatch / skipped / missing id), anti-cache probe failure, output-shape error, time-cap hang,
  or crash.
- **Consistency gate (welded, the anti-hack core)**: the harness recomputes the reference scores
  with its OWN independent implementation and requires `|cand - ref| <= score_atol` for EVERY
  scored id, with `mismatched == 0` AND `missing == 0`. You cannot trade correctness for speed.
- **Anti-cache probe**: a random subset of held-out records is perturbed (loglikelihoods reversed /
  responses reversed) and given FRESH `probe::` ids; the reference is recomputed on the perturbed
  inputs. A copied (id -> score) lookup table from a public dev run mis-scores here -> 0.
- **Harness-owned everything**: the sample set, the gold targets, the reference scorer, and the
  clock are all harness-owned. Nothing the submission reports is trusted.

## Files here
- `scoring_pipeline_ref.py` — the **STRONG BASELINE** (1.0 anchor): a competent, already-optimised
  scorer (regex precompiled once, records grouped by metric into tight typed loops, single-pass
  majority-vote). Byte-for-byte score-consistent with the reference scorer. Absorbs the cheap wins
  so a config flip can't beat it; real speedups need vectorised regex/metric compute, columnar
  scoring, or a rewritten lm_eval scoring path.
- `negative_scorer.py` — two controls: (1) `SlowButCorrectScorer` (correct but slow -> reward < 1.0,
  proves the speed gradient); (2) `SkippingScorer` (fast but skips samples -> reward 0, proves the
  welded consistency gate). Default loader returns the skipping one.

## Headroom (why the ceiling is well above 1.0)
The naive per-row Python loop (the model-visible template) is the floor. The strong baseline already
grabs regex-precompile + metric-grouping. The real ceiling comes from: fully vectorised regex
extraction over the whole column (numpy / batched string ops), a columnar majority-vote, exploiting
the lm-evaluation-harness auto-batch cache-clearing insight (PR #3654), and rewriting the scoring
path in `/app/repo` to avoid Python per-record overhead entirely. Expect a multi-x speedup ceiling
over the naive template and a clear gap above the grouped strong baseline (measured ceiling 2.24419x under the shipped in-session ABBA protocol = reward 0.5).

## ANCHOR RE-CALIBRATION RECIPE (on an H20)
1. Stage the lm-evaluation-harness repo (commit `97a5e2c7`) into the build context as `lmeval_repo/`
   (on local disk, since network filesystems are not visible to the build worker); build the image FROM
   `<internal registry>/kernelbench/wro-lmeval-base@sha256:9e14fdef` (resolve the FULL digest).
2. Copy `solution/scoring_pipeline_ref.py` -> `/opt/strong_baseline/scoring_pipeline.py` and
   `solution/negative_scorer.py` -> `/opt/negative/scoring_pipeline.py` on the scoring host (NOT baked).
3. Recompute `sha256(tests/compute_reward.py)` and write it into
   `verifier-correctness-manifest.json:compute_reward_sha256` (test.sh self-checks it).
4. Run the strong baseline through the verifier **>=5x**; set
   `strong_baseline_time_sec = median full-set wall time`. Confirm `mismatched==0` / `missing==0`
   on every held-out id (consistency 5/5).
5. Run negative (1) -> confirm reward < 1.0 AND consistency still passes (slow, not wrong). Run
   negative (2) -> confirm reward == 0 (welded consistency gate). Set `min_speedup_plausible`,
   `max_score_time_sec`, `timing_repeats` bands; confirm the noise floor << baseline time.
