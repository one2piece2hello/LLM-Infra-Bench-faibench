# fai_bench — Scoring

**English**: `SCORING.md` (this file) · **中文**: [`SCORING.zh-CN.md`](SCORING.zh-CN.md)

The authoritative implementation for each task is its own `tests/compute_reward.py`; this document describes the shape of the two reward classes so you don't misread the results. Scoring artifacts always land in `/logs/verifier/`:

```
reward.json     structured result: reward + breakdown (per-case pass counts, paired speedup, diagnostics)
reward.txt      plain text, same value as reward.json's reward
```

**The `reward` field in `reward.json` is the task's final score** — no second conversion is needed (you do not, as with some harnesses, treat `score` as a conservative value and have the leaderboard compute partial credit separately).

## The two reward classes

| Class | Tasks | Range | Shape |
|---|---|---|---|
| Performance | 77 | continuous [0, 1] | pass the correctness gate first, then score by the log speedup **relative to oracle** |
| Implementation | 8 | binary {0.0, 1.0} | all hidden cases pass and no gate fires → 1.0 |

### Performance: log speedup, **oracle is the zero point**

The 77 performance tasks all use `reward_md_log_speedup_v2_oracle_zero`:

```
speedup ≤ ref_speedup  ⇒  reward = 0
speedup > ref_speedup  ⇒  reward = min(1.0, ln(speedup / ref_speedup) / ln(ref_speedup))
                                                                        range [0, 1]
```

Three anchors determine how to read it:

- `speedup ≤ ref_speedup` (**did not exceed the oracle calibrated at authoring time**) ⇒ **reward = 0**
- `speedup == ref_speedup^1.5` ⇒ **reward = 0.5**
- `speedup ≥ ref_speedup²` ⇒ **capped at 1.0**

**Key point: tying the oracle scores 0; you must "beat the oracle" before you score anything.** This curve is a linear transform of the old curve
`r_v1 = min(1, 0.5·ln(speedup)/ln(ref_speedup))` (tying the oracle gave 0.5):

```
r_v2 = max(0, 2·r_v1 − 1)
```

The motivation is discrimination: the old curve packed a large mass of submissions near 0.5 (matching the oracle), leaving only half the range for the thing we actually want to distinguish — whether, and by how much, a submission beats the oracle. The new curve gives the entire [0,1] range to "after you've beaten the oracle."

**⚠️ New and old scores are not directly comparable.** To convert a historical score, `r_v1 = (r_v2 + 1) / 2`, and only when `r_v2 > 0` (the [0, 0.5] half of v1 is all compressed to 0 in v2 — that information is irreversible). **`ref_speedup` itself is unchanged**, so the anchors need no re-calibration.

`ref_speedup` is a constant calibrated at authoring time, hard-coded into that task's manifest under `tests/`, read-only at scoring time (never recomputed) — so a task's score is comparable across models and across time. For example, `kv-traffic-sol` has `ref_speedup = 2.5799`: a speedup of 2.58 (tie) scores **0**, ≈4.14 (`ref^1.5`) scores 0.5, ≥6.656 (`ref²`) scores 1.0.

**"speedup" is not always a wall-clock speedup** — it is the ratio of the task's declared `perf_metric`, of which there are three kinds:

| perf_metric | Meaning | Example tasks |
|---|---|---|
| wall-clock / bandwidth speedup | ABBA-paired (baseline/candidate alternating for several pairs, geometric mean over each pair's timed cases), then median across pairs | `kv-traffic-sol`, `varlen-prefill-attn-sol`, `vllm-scheduler` |
| `quality_at_fixed_budget` | quality ratio under a fixed budget, e.g. `baseline_bpb / candidate_val_bpb` | `a3-moe-train-budget`, `a4-token-efficiency-budget` |
| `quality_under_budget` | retrieval quality ratio under a fixed byte budget, e.g. at 64 B/vector, `candidate_nDCG@10 / baseline_nDCG@10` | `embed-compress-golf` (strong baseline nDCG@10 = 0.459151, ref = 1.4290) |

**ABBA pairing** is the key technique in performance measurement: baseline and candidate are measured alternately in pairs, the ratio is taken per pair, then the median across pairs. This way machine noise, warm-up effects, and frequency drift act equally on both sides and are not counted as speedup.

### Implementation: binary, any single failure → 0

The 8 implementation tasks have rewards of only 0.0 and 1.0:

```
reward = 1.0  if and only if  all hidden cases pass  and  no cheat/forbidden-edit gate fires
reward = 0.0  in every other case
```

The number of cases varies by task, from a dozen to over a hundred hidden cases/gates (the actual count per task is decided by that task's `tests/`). **Per-case pass counts and breakdown diagnostics are still written into `reward.json`, but the score is never moved out of {0.0, 1.0}** — they are for offline analysis only.

Some implementation tasks **carry timing measurements**, but the timing **does not enter the score** — it is only a diagnostic, or a **precondition gate** ("must clear a strong baseline," e.g. requiring the paired-ratio median of several hidden workloads to be > 1.0 and non-degenerate). These tasks are still binary: all gates pass = 1.0, otherwise 0.0 — **seeing a `speedup` field in `reward.json` does not make it a performance task**; go by `reward_class` / `reward_formula`.

## Zeroing gates (hard fail)

Any of the following, once hit, sets the task **reward = 0** regardless of the measurement:

1. **Build/import/readiness failure** — the submitted code doesn't start
2. **Any correctness case fails** — the correctness gate for a performance task is all-or-nothing, no partial credit
3. **Cheat detected** — the frozen surface was tampered with, paired ratios are identically equal (fabricated measurement), or the speedup is physically implausible
4. **Touched `forbidden_edit_paths`** — paths listed in `task.toml` that are sha256-frozen
5. **Performance task with `speedup ≤ 1`** — not beating the strong baseline is no improvement at all
6. **`ref_speedup` missing or ≤ 1** — refuse to score when the anchor is untrustworthy, rather than emit a suspect score

**Distinguish "zeroing gate" from "the curve gives 0":** `1 < speedup ≤ ref_speedup` (beat the strong baseline but didn't exceed the oracle) is **not** a hard fail —
it is **the curve itself evaluating to 0**, with `hard_fail_reasons` left empty. The semantics of `hard_fail` are "this run is invalid / cheating," not "the score is low"; when reading results, `reward = 0` with an empty `hard_fail_reasons` means "ran fine, just didn't beat the oracle."

Anti-cheat is not only these gates: before scoring, each task's `tests/test.sh` also does a source scan (forbidding references to verifier-internal paths like `/tests/`, `compute_reward`, `reward.json`), forces a rebuild from source when necessary, and does symbol-level checks (`ldd`/`nm` to see whether it secretly linked the original library).

## Submission budget: decided by two dimensions [subset × task type]

| Subset | Performance | Implementation |
|---|---|---|
| `kfc` (55 tasks) | **1** | **1** |
| `lh` (20 tasks) | 1–16 | 1 |
| `e2e` (10 tasks) | 1–16 | 1 |

**The entire `kfc` subset is single-submission, regardless of task type** — all 55 have exactly one scoring opportunity, in one of two forms:

- **50 tasks** ship the loop harness but with `MIN_SUBMISSIONS=MAX_SUBMISSIONS=1`: the first `bash /opt/loop/submit.sh` scores and then **finalizes immediately**, with no second scored attempt (calling it again just re-finalizes the same recorded snapshot, without scoring new changes)
- **5 tasks** (`chunked-mlp-recompute`, `ckpt-dcp-meta-bbox-merge`, `mamba-zoh-discretize`, `s4-fft-longconv`, `wre-verl-grpo-advantage-loop16`) have no `submit.sh`: changes are left in the working tree and scored once by `tests/test.sh` mounted after the session ends

Only the **`lh`/`e2e` performance tasks** — **26 in total** — run the 1–16 round protocol. The cap of 16 is **not a hard requirement**: the agent decides when to `submit.sh --finalize` and stop (auto-finalizes at k=16), and need not fill the quota. Implementation tasks are single-submission in every subset.

Each task's budget is declared in **three places** and must agree: `MIN_SUBMISSIONS`/`MAX_SUBMISSIONS` in `environment/loop/submit.sh`, `environment/loop/private/manifest.json`, and the `[loop]` section of `task.toml`. Any disagreement among the three is a package defect.

## loop16 tasks score the "best round," not the last edit

For those 26 tasks running 1–16 rounds, `submit.sh --finalize` implants the **historically best round** pointed to by `/logs/loop/best.json` as the scored artifact. This means:

- if the agent's last edit is worse than something mid-run, it **does not affect the score**
- per-round measurements are in `/logs/loop/state.jsonl`, the round count in `/logs/loop/count`
- if the session is cut off by a timeout, `--finalize` still implants the best round so far — so **a nonzero reward does not prove the session ended cleanly**. To judge whether the task was fully answered, look at whether the session ended normally, not at whether reward > 0

**A known behavior under the new curve**: `best.json` is updated by strictly increasing dev reward. When a session **never beats the oracle at any point**, every round's dev reward is 0, so the "best round" stays at round 1 — the `best_so_far` feedback and the finally-implanted artifact are both the earliest tree, not the one with the highest speedup. **This does not affect the score** (below oracle throughout, whichever round is implanted the result is 0), but note that the finalized/implanted artifact in that case is that earliest round, not the highest-speedup one. The agent's progress signal is unaffected: each round's feedback still gives `dev_speedup`, so it sees the improvement from 1.05× → 1.99×, only the reward stays 0.

## Aggregating to the bench level

`tasks_index.json` gives each task's `category` and `medium_topic`/`big_topic`. When comparing models:

- do **not** directly average performance and implementation classes together — the former is continuous, the latter binary, and mixing lets the implementation 1.0s drown out the differences among performance tasks
- **cross-version comparison must confirm both sides use the same reward curve** — check `reward.json`'s `schema_version` (the v2 curve is `kernelbench_reward_v3_oracle_relative`); scores from different curves are not directly comparable
