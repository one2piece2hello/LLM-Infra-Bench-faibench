"""e2e-h3-eval-harness-throughput-quality verifier (family C, quality-under-budget; H3 LLM 评测与压测).

Optimize the END-TO-END THROUGHPUT of an LLM evaluation harness's scoring/aggregation pipeline
(the lm-evaluation-harness filter-ensemble + metric path: regex answer extraction, take-first /
majority-vote response transforms, loglikelihood aggregation, per-doc metric compute), on a FIXED
hidden evaluation set, WITHOUT changing a single per-sample score. Speed for free is the whole
point; speed by skipping / approximating samples is foreclosed by a hard EXACT-MATCH consistency
gate.

Single-shot, eval-only, DETERMINISTIC. The candidate submits a scoring pipeline exposing
    `load_scoring_pipeline_for_verification(device)` -> object with
        `.score(samples: list[dict]) -> list[dict]`
where each input sample carries the FROZEN, harness-produced model outputs for one eval instance
(generated text and/or per-choice loglikelihoods, the requested filter pipeline, the metric name,
and the gold target) and each output row must be `{"id": <sample id>, "score": <float>}`.

The harness:
  * feeds ITS OWN held-out sample set (frozen model outputs the candidate never generated);
  * TIMES the candidate's `.score()` over the full set, repeated `timing_repeats` times, median;
  * TIMES the calibrated STRONG BASELINE scorer the same way (harness-owned reference impl);
  * RECOMPUTES the reference per-sample scores with its OWN independent implementation;
  * requires EXACT per-sample agreement (|cand - ref| <= score_atol for every scored id, and NO
    id skipped, dropped, added, or reordered-away) -> the consistency gate is welded shut;
  * ABBA-pairs the candidate against the reviewer-owned STRONG BASELINE re-measured in-session
    (>= 5 alternating pairs; speedup = median(baseline_time / candidate_time));
  * emits reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 in [0,1] per the bench reward spec
    (ref_speedup is a frozen AUTHORING-TIME constant; the oracle is never run at scoring).
    Merely MATCHING the strong baseline is speedup == 1.0 and therefore reward 0.
  * [history, 2026-07-27] the superseded form was unbounded (baseline_time/candidate_time with
    1.0 meaning "matched the baseline"); it is gone, and so is the baked /opt/verifier fallback
    copy of this file that still implemented it.

Nothing the submission reports about correctness or timing is trusted: the harness owns the sample
set, the gold targets, the reference scores, and the clock. A cached (id -> score) table copied
from a public dev run is useless: held-out ids + held-out gold + the harness re-times real work and
an anti-cache probe perturbs inputs so a lookup table produces WRONG scores and fails the gate.

Thresholds, the ref_speedup constant and the held-out-workload digest are read from the FROZEN
manifest, which -- like the held-out sample set itself -- ships ONLY with the fresh /tests upload
and is NOT baked into the task image.
"""
from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))


# =============================================================================
# TRUST BOUNDARY (added 2026-07-28, vacuity sweep V7 — MEASURED, not theoretical)
# =============================================================================
# `import_solution()` loads the candidate with importlib INTO THIS PROCESS: the
# process that owns the clock, the held-out sample set, the INDEPENDENT reference
# scorer, the pass/fail accumulator and the reward function.  Everything this
# module trusts is therefore a module attribute the candidate can rewrite at
# import time.  Measured on the pinned image `-val:v1@sha256:fdf98e7d…`
# (honest controls in the same run
# scored 0.0 at speedup 0.9522 / 0.9260):
#
#   * a candidate BYTE-FOR-BYTE identical to the strong baseline + 12 lines that
#     book the real time spent inside its own .score() as "clock debt" and
#     subtract it from every later `time.perf_counter()` read scored
#     reward 0.9477371018832937 at speedup 4.6283 (K=5.5), and 0.5402572463687533
#     at speedup 2.3951 (K=2.6 — i.e. tuned to be indistinguishable from the
#     oracle).  The patch is SELECTIVE (a global slowdown is useless because the
#     baseline shares the clock) and IDEMPOTENT (it never compounds), so no
#     plausibility bound was available to catch it — and `min_speedup_plausible`
#     is a FLOOR only, there is no ceiling.
#   * `sys.modules["__main__"].compute_log_reward = lambda *a, **k: 1.0` plus a
#     `Check.__init__` that forces `passed=True, hard=False` scored reward 1.0
#     on a candidate SLOWER than the baseline (emitted speedup 0.9290) and 1.0
#     on the sample-SKIPPING negative control (which scores 0.0 honestly).
#   * `sys.modules["__main__"]._reference_score_one = lambda s: 0.0` rewrote the
#     harness's own ground truth, so a scorer whose .score() returns 0.0 for
#     every row — it never reads a single field — passed the welded consistency
#     gate (matched 20000 / mismatched 0), passed the anti-cache probe, and
#     scored reward 1.0 at speedup 15.0002.
#
# None of those probes contains a token that test.sh step-3b's SUSPICIOUS grep
# looks for, so the source scan is no defence.
#
# The fix, mirroring the proven pattern on the `-sol` tasks: bind every
# primitive the measurement depends on to its ORIGINAL object BEFORE any
# candidate module can be imported, and re-assert identity at every point where
# a result crosses into the score.  A replacement is reward.md 前置门 3 (作弊:
# modifying the bench harness) -> HARD 0 with the named reason
# `cheating_harness_tamper`.
_TAMPER_REASON = "cheating_harness_tamper"


class HarnessTamper(RuntimeError):
    """Raised when a measurement primitive is no longer the object we bound."""


# Module globals whose identity gates the score.  Populated by
# _snapshot_primitives() at the very bottom of this file (after every def), which
# still runs long before import_solution() can execute candidate code.
_GUARDED_GLOBALS: tuple[str, ...] = (
    # the clock and the timing loop
    "_time_scoring", "_cv",
    # the harness-owned INDEPENDENT reference scorer (the ground truth)
    "_normalise", "_apply_filter", "_reference_score_one", "reference_scores", "_WS",
    # the held-out workload + the anti-cache probe
    "load_heldout", "_read_jsonl", "_perturb_for_anticache",
    # the comparison / accumulator
    "_as_score_map", "_consistency", "Check", "result",
    # the reward itself
    "compute_log_reward", "_validate_ref_speedup", "leaderboard_metrics",
    "_hard_fail_reasons", "write_outputs", "markdown_report",
    # loading + path hygiene
    "import_solution", "call_pipeline_loader", "sanitize_python_path",
    "is_regular_workspace_file", "check_required_files", "check_validation", "run_all",
    "load_manifest", "_frozen",
    # the guard itself
    "_assert_pristine", "_snapshot_primitives",
)
_ORIG_GLOBALS: dict[str, Any] = {}
_ORIG_EXTERNAL: tuple[tuple[str, Any, Any], ...] = ()
_ORIG_PRIM: dict[str, Any] = {}          # name -> the original callable (tamper-path use)
# reward-relevant manifest values, frozen right after load_manifest() and before
# any candidate code runs, so mutating the live cfg dict cannot move the reward
# (e.g. ref_speedup -> 1.0000001 makes ln(ref_speedup) ~ 0 and caps the reward).
_CFG_GUARDED_KEYS: tuple[str, ...] = (
    "ref_speedup", "score_atol", "min_consistency_fraction", "abba_pairs",
    "timing_repeats", "min_speedup_plausible", "anti_cache_perturb_frac",
    "probe_seed", "max_samples", "heldout_samples_sha256",
)
_ORIG_CFG: dict[str, Any] = {}


def _snapshot_primitives() -> None:
    """Bind the originals.  Called at module bottom — before any candidate import."""
    global _ORIG_EXTERNAL
    g = globals()
    for name in _GUARDED_GLOBALS:
        if name in g:
            _ORIG_GLOBALS[name] = g[name]
    _bi = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    _ORIG_EXTERNAL = (
        ("time.perf_counter", lambda: time.perf_counter, time.perf_counter),
        ("time.monotonic", lambda: time.monotonic, time.monotonic),
        ("time.time", lambda: time.time, time.time),
        ("statistics.median", lambda: statistics.median, statistics.median),
        ("statistics.fmean", lambda: statistics.fmean, statistics.fmean),
        ("statistics.stdev", lambda: statistics.stdev, statistics.stdev),
        ("math.log", lambda: math.log, math.log),
        ("math.isfinite", lambda: math.isfinite, math.isfinite),
        ("hashlib.sha256", lambda: hashlib.sha256, hashlib.sha256),
        ("json.dumps", lambda: json.dumps, json.dumps),
        ("json.loads", lambda: json.loads, json.loads),
        ("random.Random", lambda: random.Random, random.Random),
        ("inspect.signature", lambda: inspect.signature, inspect.signature),
        ("_re.compile", lambda: _re.compile, _re.compile),
        ("Check.__init__", lambda: Check.__init__, Check.__init__),
        ("Path.read_text", lambda: Path.read_text, Path.read_text),
        ("Path.read_bytes", lambda: Path.read_bytes, Path.read_bytes),
        ("Path.write_text", lambda: Path.write_text, Path.write_text),
        ("Path.exists", lambda: Path.exists, Path.exists),
        ("builtins.abs", lambda: _bi.get("abs"), abs),
        ("builtins.float", lambda: _bi.get("float"), float),
        ("builtins.len", lambda: _bi.get("len"), len),
        ("builtins.open", lambda: _bi.get("open"), open),
        ("builtins.sorted", lambda: _bi.get("sorted"), sorted),
        ("builtins.max", lambda: _bi.get("max"), max),
    )
    _ORIG_PRIM.clear()
    for _n, _g, _o in _ORIG_EXTERNAL:
        _ORIG_PRIM[_n] = _o


def _snapshot_cfg(cfg: dict[str, Any]) -> None:
    """Freeze the reward-relevant manifest values before any candidate code runs."""
    _ORIG_CFG.clear()
    for k in _CFG_GUARDED_KEYS:
        _ORIG_CFG[k] = cfg.get(k)


def _assert_pristine(tag: str, cfg: dict[str, Any] | None = None) -> None:
    """reward.md 前置门 3.  Any replacement of a primitive the measurement depends
    on — the clock, the reference scorer, the comparison, the accumulator, the
    reward function, the output writer — is a HARD FAIL with a named reason."""
    bad: list[str] = []
    g = globals()
    for name, orig in _ORIG_GLOBALS.items():
        if g.get(name) is not orig:
            bad.append(name)
    for name, getter, orig in _ORIG_EXTERNAL:
        try:
            if getter() is not orig:
                bad.append(name)
        except Exception:  # noqa: BLE001
            bad.append(name + "(unreadable)")
    if cfg is not None and _ORIG_CFG:
        for k, want in _ORIG_CFG.items():
            if cfg.get(k) != want:
                bad.append("manifest:" + k)
    if bad:
        raise HarnessTamper(
            "harness_tamper_detected[%s]: the candidate replaced the harness "
            "measurement primitive(s) %s. The clock, the held-out workload, the "
            "INDEPENDENT reference scorer, the consistency comparison, the "
            "pass/fail accumulator and the reward function are the evaluation "
            "surface; rewriting them in the process that grades you is cheating "
            "(reward.md 前置门 3)." % (tag, ", ".join(sorted(set(bad)))))


def _emit_tamper_contract(detail: str) -> None:
    """Write the FULL 6-file contract with reward 0.0 and a NAMED reason, using
    only the ORIGINAL primitives so a patched writer cannot intercept it."""
    _dumps = _ORIG_PRIM.get("json.dumps", json.dumps)
    _wtext = _ORIG_PRIM.get("Path.write_text", Path.write_text)
    _exists = _ORIG_PRIM.get("Path.exists", Path.exists)
    zero = {"task_type": "performance", "reward": 0.0, "partial_score": 0.0,
            "binary_pass": 0, "all_hard_gates_pass": False,
            "hard_fail_reasons": [_TAMPER_REASON], "speedup": None,
            "ref_speedup": None, "cv": {}, "passed": 0, "total": 0,
            "detail": detail}
    vdir = Path("/logs/verifier")
    try:
        present = bool(_exists(vdir))
    except Exception:  # noqa: BLE001
        present = False
    if present:
        for name, body in (
            ("metrics.json", zero),
            ("reward.json", {k: zero[k] for k in ("task_type", "reward",
                                                  "hard_fail_reasons", "speedup",
                                                  "ref_speedup", "cv", "binary_pass")}),
            ("verifier_state.json", zero),
            ("correctness_results.json", {"binary_pass": 0, "all_hard_gates_pass": False,
                                          "passed": 0, "total": 0, "detail": detail}),
            ("benchmark_results.json", {"score_metric": "log_speedup_vs_ref_speedup",
                                        "reward": 0.0, "speedup": None,
                                        "ref_speedup": None,
                                        "metric_void_on_hard_fail": True,
                                        "detail": detail}),
        ):
            _wtext(vdir / name, _dumps(body, indent=2, default=str) + "\n",
                   encoding="utf-8")
        _wtext(vdir / "reward.txt", "0.000000\n", encoding="utf-8")
    print("HARD FAIL (%s): %s" % (_TAMPER_REASON, detail))


def _frozen(name: str, env: str | None = None) -> Path:
    """Resolve a frozen-surface file.

    🔴 2026-07-27: the ONLY search root is the FRESH /tests upload.  The pre-2026-07-27
    version also fell back to a baked /opt/verifier copy; that copy carried the held-out
    sample set, the pre-migration unbounded scorer and the calibration manifest INSIDE the
    task image, where a root solver could read all three (the scoring container runs as uid 0, so root-0700 is
    no protection at all).  /opt/verifier has been removed from the image and is no longer
    consulted, so a missing /tests upload now fails CLOSED instead of silently scoring with
    stale semantics against a solver-visible sample set.
    """
    if env and os.environ.get(env):
        return Path(os.environ[env])
    return Path("/tests") / name


MANIFEST_PATH = _frozen("verifier-correctness-manifest.json", "E2E_MANIFEST_PATH")
HELDOUT_SAMPLES_PATH = _frozen("heldout_samples.jsonl", "E2E_SAMPLES_PATH")
# The STRONG-BASELINE implementation the candidate is paired against (ABBA).  Reviewer-owned,
# uploaded fresh under /tests/oracles, NEVER baked into the image (MOD_SPEC 改动 3).
BASELINE_IMPL_PATH = _frozen("oracles/strong_baseline_scoring_pipeline.py", "E2E_BASELINE_IMPL")

_DEFAULTS = {
    "score_atol": 1e-9,               # exact-match tolerance per sample (welded consistency gate)
    "min_consistency_fraction": 1.0,  # ALL scored ids must match exactly (no skipping for speed)
    "strong_baseline_time_sec": None, # DIAGNOSTIC ONLY since 2026-07-27: the speedup denominator is
                                      # now the baseline RE-MEASURED in-session inside each ABBA pair
    "ref_speedup": None,              # reward.md: oracle median ABBA speedup (frozen constant)
    "abba_pairs": 5,                  # reward.md: >= 5 alternating baseline/candidate pairs
    "timing_repeats": 5,              # repeat the timed scoring pass; take the median
    "min_speedup_plausible": 0.05,    # anti-noise floor: absurdly small speedup -> treat as broken/0
    "max_score_time_sec": 900,        # a single scoring pass may not exceed this (anti-hang)
    "anti_cache_perturb_frac": 0.25,  # fraction of samples perturbed for the anti-cache probe
    "probe_seed": 20240725,
    "max_samples": 200000,
    "heldout_samples_sha256": None,   # digest of the CALIBRATED timed workload (frozen manifest)
}


def load_manifest() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        if MANIFEST_PATH.exists():
            m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            th = m.get("thresholds", m)
            for k in _DEFAULTS:
                if k in th and th[k] is not None:
                    cfg[k] = th[k]
    except Exception:
        pass
    return cfg


class Check:
    def __init__(self, name: str, passed: bool, message: str, details: dict[str, Any] | None = None, hard: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.hard = hard


def result(name: str, passed: bool, message: str, hard: bool = True, **details: Any) -> Check:
    return Check(name, passed, message, details, hard)


def sanitize_python_path(workspace: Path) -> None:
    blocked = {workspace.resolve(), Path.cwd().resolve()}
    clean: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved in blocked:
            continue
        clean.append(entry)
    sys.path[:] = clean


def is_regular_workspace_file(path: Path, workspace: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_symlink():
        return False, "symlinks are not accepted"
    if not path.is_file():
        return False, "not a regular file"
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False, "file resolves outside the submission dir"
    return True, "present"


def import_solution(path: Path, mod_name: str = "submitted_scorer"):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    workspace_str = str(path.parent.resolve())
    inserted = False
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(workspace_str)
            except ValueError:
                pass
    return module


def _read_jsonl(path: Path, cap: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("# provenance-marker"):
            continue
        rows.append(json.loads(line))
        if len(rows) >= cap:
            break
    return rows


def load_heldout(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not HELDOUT_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Held-out samples not found at {HELDOUT_SAMPLES_PATH}")
    # 🔴 The timed workload must be BYTE-IDENTICAL to the one ref_speedup was calibrated on,
    # otherwise the log-reward denominator no longer refers to the same benchmark.  Since the
    # set now ships with the fresh /tests upload (it used to be baked root-0700 into the image,
    # i.e. solver-readable), assert its digest against the frozen manifest constant.
    want = cfg.get("heldout_samples_sha256")
    if want:
        got = hashlib.sha256(HELDOUT_SAMPLES_PATH.read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(
                "held-out sample set digest mismatch: the timed workload is not the one "
                f"ref_speedup was calibrated on (expected {want}, got {got})")
    return _read_jsonl(HELDOUT_SAMPLES_PATH, int(cfg["max_samples"]))


def call_pipeline_loader(module: Any, device: str):
    loader = getattr(module, "load_scoring_pipeline_for_verification", None)
    if loader is None:
        raise RuntimeError(
            "submission must define load_scoring_pipeline_for_verification(device) returning an "
            "object exposing .score(samples) -> list[{'id','score'}]."
        )
    sig = inspect.signature(loader)
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for name, parameter in sig.parameters.items():
        if name == "device":
            kwargs[name] = device
        elif parameter.default is inspect._empty and not positional:
            positional.append(device)
    pipe = loader(*positional, **kwargs)
    if pipe is None:
        raise RuntimeError("load_scoring_pipeline_for_verification returned None")
    return pipe


# -----------------------------------------------------------------------------
# HARNESS-OWNED reference scorer: an INDEPENDENT implementation of the lm-eval
# filter + metric path (regex extraction, take-first / majority-vote transforms,
# loglikelihood-choice acc, exact-match). The candidate must reproduce THESE scores
# EXACTLY while being faster. This function is the ground truth; it is never trusted
# to come from the repo or the submission.
# -----------------------------------------------------------------------------
import re as _re

_WS = _re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def _apply_filter(raw: Any, filt: str, pattern: str | None) -> str:
    """Response transform: regex extraction then a take-first / majority selection."""
    if isinstance(raw, list):
        cands = [str(x) for x in raw]
    else:
        cands = [str(raw)]
    if pattern:
        rx = _re.compile(pattern)
        extracted: list[str] = []
        for c in cands:
            m = rx.search(c)
            extracted.append(m.group(1) if (m and m.groups()) else (m.group(0) if m else ""))
        cands = extracted
    if filt == "take_first":
        return cands[0] if cands else ""
    if filt == "majority_vote":
        counts: dict[str, int] = {}
        for c in cands:
            key = _normalise(c)
            counts[key] = counts.get(key, 0) + 1
        # deterministic: highest count, then lexicographically smallest key
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0] if best else ""
    return cands[0] if cands else ""


def _reference_score_one(sample: dict[str, Any]) -> float:
    metric = str(sample.get("metric", "exact_match"))
    if metric == "loglikelihood_acc":
        # multiple-choice: argmax over per-choice loglikelihoods; 1.0 if == gold index.
        lls = [float(x) for x in sample.get("choice_loglikelihoods", [])]
        if not lls:
            return 0.0
        pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
        return 1.0 if pred == int(sample.get("gold_index", -1)) else 0.0
    # generative metrics: run the filter pipeline then compare to gold text
    pred = _apply_filter(sample.get("response"), str(sample.get("filter", "take_first")),
                         sample.get("filter_pattern"))
    gold = str(sample.get("gold", ""))
    if metric == "exact_match":
        return 1.0 if _normalise(pred) == _normalise(gold) else 0.0
    if metric == "contains":
        return 1.0 if _normalise(gold) in _normalise(pred) else 0.0
    if metric == "prefix_match":
        return 1.0 if _normalise(pred).startswith(_normalise(gold)) else 0.0
    return 0.0


def reference_scores(samples: list[dict[str, Any]]) -> dict[str, float]:
    return {str(s["id"]): _reference_score_one(s) for s in samples}


def _perturb_for_anticache(samples: list[dict[str, Any]], frac: float, seed: int) -> list[dict[str, Any]]:
    """Anti-cache probe: rewrite the response/loglikelihoods of a random subset so a copied
    (id -> score) lookup table produces WRONG scores. The perturbed inputs get FRESH ids so the
    reference is recomputed for them; a genuine scorer that actually reads the inputs still matches."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    n = len(samples)
    idx = set(rng.sample(range(n), max(1, int(n * frac)))) if n else set()
    for i, s in enumerate(samples):
        t = dict(s)
        t["id"] = f"probe::{s['id']}"
        if i in idx:
            if t.get("metric") == "loglikelihood_acc" and t.get("choice_loglikelihoods"):
                lls = [float(x) for x in t["choice_loglikelihoods"]]
                lls = list(reversed(lls))  # flips the argmax deterministically
                t["choice_loglikelihoods"] = lls
            else:
                r = t.get("response")
                if isinstance(r, list):
                    t["response"] = [str(x)[::-1] for x in r]
                else:
                    t["response"] = str(r)[::-1]
        out.append(t)
    return out


def _time_scoring(pipe: Any, samples: list[dict[str, Any]], repeats: int,
                  max_sec: float) -> tuple[list[dict[str, Any]], float, bool]:
    """Time pipe.score(samples) `repeats` times; return (last output, median seconds, ok).

    🔴 V7: `_perf` / `_median` are the ORIGINAL objects, resolved from the snapshot rather
    than by module-attribute lookup, so a `time.perf_counter = ...` patch installed by the
    candidate at import time cannot reach the sampling points.  _assert_pristine() then
    turns the attempt itself into a hard fail (the patch is still DETECTED, not merely
    routed around — a silently-ignored patch would let the cheat retry for free)."""
    _perf = _ORIG_PRIM.get("time.perf_counter", time.perf_counter)
    _median = _ORIG_PRIM.get("statistics.median", statistics.median)
    times: list[float] = []
    last_out: list[dict[str, Any]] = []
    for _ in range(max(1, repeats)):
        _assert_pristine("timed pass (entry)")
        t0 = _perf()
        out = pipe.score([dict(s) for s in samples])
        dt = _perf() - t0
        _assert_pristine("timed pass (exit)")
        if dt > max_sec:
            return list(out) if isinstance(out, list) else [], dt, False
        times.append(dt)
        last_out = list(out) if isinstance(out, list) else []
    times.sort()
    return last_out, times[len(times) // 2], True


def _as_score_map(out: Any) -> dict[str, float] | None:
    if not isinstance(out, (list, tuple)):
        return None
    m: dict[str, float] = {}
    for row in out:
        if not isinstance(row, dict) or "id" not in row or "score" not in row:
            return None
        try:
            m[str(row["id"])] = float(row["score"])
        except Exception:
            return None
    return m


def _consistency(cand: dict[str, float], ref: dict[str, float], atol: float) -> tuple[int, int, int]:
    """Return (matched, mismatched, missing) over the reference id set."""
    matched = mismatched = missing = 0
    for rid, rv in ref.items():
        if rid not in cand:
            missing += 1
            continue
        if math.isfinite(cand[rid]) and abs(cand[rid] - rv) <= atol:
            matched += 1
        else:
            mismatched += 1
    return matched, mismatched, missing


def check_required_files(workspace: Path) -> list[Check]:
    path = workspace / "scoring_pipeline.py"
    passed, message = is_regular_workspace_file(path, workspace)
    return [result("Required file: scoring_pipeline.py", passed, message)]


def check_validation(workspace: Path, cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    try:
        import torch  # noqa: F401
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # 🔴 V7 trust boundary: the LAST point at which nothing candidate-authored has
    # executed in this process.  Everything after import_solution() is inside the
    # blast radius, so re-assert at every step where a value crosses into the score.
    _assert_pristine("pre-import", cfg)
    module = import_solution(workspace / "scoring_pipeline.py")
    _assert_pristine("post-import", cfg)
    pipe = call_pipeline_loader(module, device)
    _assert_pristine("post-loader", cfg)
    if not (hasattr(pipe, "score") and callable(getattr(pipe, "score"))):
        checks.append(result("Pipeline exposes .score()", False, "pipeline has no callable .score"))
        return checks

    samples = load_heldout(cfg)
    ref = reference_scores(samples)
    _assert_pristine("post-reference", cfg)
    atol = float(cfg["score_atol"])
    repeats = int(cfg["timing_repeats"])
    max_sec = float(cfg["max_score_time_sec"])

    # --- timed correctness pass on the real held-out set ---
    cand_out, cand_time, ok = _time_scoring(pipe, samples, repeats, max_sec)
    checks.append(result("Candidate scoring completes within time cap", ok,
                         f"median scoring time={cand_time:.4f}s (cap {max_sec:.0f}s)" if ok
                         else f"scoring pass exceeded cap {max_sec:.0f}s (hang/degenerate)",
                         candidate_time_sec=cand_time))
    if not ok:
        return checks

    cand_map = _as_score_map(cand_out)
    if cand_map is None:
        checks.append(result("Candidate .score output shape", False,
                             "expected list of {'id','score'} rows"))
        return checks

    matched, mismatched, missing = _consistency(cand_map, ref, atol)
    _assert_pristine("post-consistency", cfg)
    total_ref = len(ref)
    extra = len(set(cand_map) - set(ref))
    frac = (matched / total_ref) if total_ref else 0.0
    min_frac = float(cfg["min_consistency_fraction"])
    consistent = (mismatched == 0 and missing == 0 and frac >= min_frac)
    checks.append(result("Per-sample exact-match consistency gate (welded)", consistent,
                         f"matched={matched}/{total_ref} mismatched={mismatched} missing={missing} "
                         f"extra_ids={extra} (require ALL exact; atol={atol:.1e})",
                         matched=matched, mismatched=mismatched, missing=missing,
                         extra_ids=extra, consistency_fraction=frac))
    if not consistent:
        return checks

    # --- anti-cache probe: perturbed inputs with fresh ids; a lookup table now mis-scores. ---
    probe_samples = _perturb_for_anticache(samples, float(cfg["anti_cache_perturb_frac"]),
                                            int(cfg["probe_seed"]))
    probe_ref = reference_scores(probe_samples)
    probe_out = pipe.score([dict(s) for s in probe_samples])
    probe_map = _as_score_map(probe_out)
    if probe_map is None:
        checks.append(result("Anti-cache probe output shape", False, "probe .score output malformed"))
        return checks
    p_match, p_mis, p_miss = _consistency(probe_map, probe_ref, atol)
    _assert_pristine("post-anti-cache-probe", cfg)
    probe_ok = (p_mis == 0 and p_miss == 0)
    checks.append(result("Anti-cache probe (perturbed inputs re-scored correctly)", probe_ok,
                         f"probe matched={p_match}/{len(probe_ref)} mismatched={p_mis} missing={p_miss} "
                         "(a copied id->score table fails here)",
                         probe_matched=p_match, probe_mismatched=p_mis, probe_missing=p_miss))
    if not probe_ok:
        return checks

    # --- reward.md gate 6: the frozen ref_speedup constant must exist and be > 1 ---
    ref_speedup = cfg.get("ref_speedup")
    ref_ok, ref_msg = _validate_ref_speedup(ref_speedup)
    checks.append(result("ref_speedup constant calibrated (reward.md gate 6)", ref_ok, ref_msg,
                         ref_speedup=ref_speedup))
    if not ref_ok:
        return checks

    # --- ABBA pairing: the STRONG BASELINE is RE-MEASURED in-session, alternating with the
    #     candidate, for >= 5 pairs (reward.md: speedup = median(baseline_ms / candidate_ms)).
    #     The old design divided by a cross-host CONSTANT (strong_baseline_time_sec); that is now
    #     a diagnostic only. ---
    if not BASELINE_IMPL_PATH.exists():
        checks.append(result("Frozen surface: strong-baseline implementation present", False,
                             f"{BASELINE_IMPL_PATH} missing — the ABBA pairing partner ships with "
                             "the fresh /tests upload and is required to measure a speedup"))
        return checks
    try:
        base_module = import_solution(BASELINE_IMPL_PATH, mod_name="strong_baseline_scorer")
        base_pipe = call_pipeline_loader(base_module, device)
    except Exception as exc:  # noqa: BLE001
        checks.append(result("Strong-baseline implementation loads", False,
                             f"{type(exc).__name__}: {exc}"))
        return checks

    n_pairs = max(5, int(cfg.get("abba_pairs") or 5))
    _median = _ORIG_PRIM.get("statistics.median", statistics.median)
    ratios: list[float] = []
    base_times: list[float] = []
    cand_times: list[float] = []
    for i in range(n_pairs):
        _assert_pristine("ABBA pair %d" % i, cfg)
        # alternate the within-pair order so a monotone drift cannot bias one side: BA / AB / …
        if i % 2 == 0:
            _o, bt, bok = _time_scoring(base_pipe, samples, 1, max_sec)
            _o, ct, cok = _time_scoring(pipe, samples, 1, max_sec)
        else:
            _o, ct, cok = _time_scoring(pipe, samples, 1, max_sec)
            _o, bt, bok = _time_scoring(base_pipe, samples, 1, max_sec)
        if not (bok and cok) or ct <= 0 or bt <= 0:
            checks.append(result("ABBA pair completes within the time cap", False,
                                 f"pair {i}: baseline_ok={bok} candidate_ok={cok} "
                                 f"baseline={bt:.4f}s candidate={ct:.4f}s"))
            return checks
        base_times.append(bt)
        cand_times.append(ct)
        ratios.append(bt / ct)
    _assert_pristine("post-ABBA", cfg)
    speedup = _median(ratios)
    cv = {"baseline": _cv(base_times), "candidate": _cv(cand_times)}

    # --- gate 3: degenerate identity paired ratios ---
    identity = all(abs(r - 1.0) < 1e-12 for r in ratios)
    checks.append(result("Paired ratios are not a degenerate identity", not identity,
                         "paired ratios are all exactly 1.0 — the candidate is the baseline / the "
                         "timing is faked" if identity
                         else f"{len(ratios)} distinct ABBA ratios measured"))
    if identity:
        return checks

    min_speedup = float(cfg["min_speedup_plausible"])
    plausible = math.isfinite(speedup) and speedup >= min_speedup
    checks.append(result("Speedup plausible (anti-noise)", plausible,
                         f"speedup={speedup:.4f}x (median of {len(ratios)} ABBA pairs) "
                         f">= floor {min_speedup:.2f}" if plausible
                         else f"speedup={speedup:.4f}x below floor {min_speedup:.2f}",
                         hard=False, speedup=speedup,
                         abba_pairs=len(ratios),
                         abba_pair_ratios=[round(r, 6) for r in ratios],
                         baseline_time_sec=round(statistics.median(base_times), 6),
                         candidate_time_sec=round(statistics.median(cand_times), 6),
                         calibrated_baseline_time_sec_diagnostic=cfg.get("strong_baseline_time_sec"),
                         cv_baseline=cv["baseline"], cv_candidate=cv["candidate"],
                         ref_speedup=float(ref_speedup)))

    # --- reward.md gate 5: did not cross the strong baseline ---
    crossed = speedup > 1.0
    checks.append(result("Crossed the strong baseline (reward.md gate 5)", crossed,
                         f"speedup={speedup:.4f}x > 1.0" if crossed
                         else f"speedup={speedup:.4f}x <= 1.0 — did not beat the strong baseline",
                         speedup=speedup))
    return checks


def run_all(workspace: Path, cfg: dict[str, Any]) -> dict[str, list[Check]]:
    checks: dict[str, list[Check]] = {"required_files": check_required_files(workspace)}
    if not all(c.passed for c in checks["required_files"]):
        return checks
    checks["validation"] = check_validation(workspace, cfg)
    return checks


def markdown_report(checks: dict[str, list[Check]]) -> str:
    total = sum(len(group) for group in checks.values())
    passed = sum(1 for group in checks.values() for c in group if c.passed)
    lines = ["# Verification Report\n\n", "## Summary\n\n",
             f"- Total checks: {total}\n", f"- Passed: {passed}\n", f"- Failed: {total - passed}\n\n"]
    for category, group in checks.items():
        lines.append(f"## {category.replace('_', ' ').title()}\n\n")
        for c in group:
            status = "PASS" if c.passed else "FAIL"
            details = f" Details: {json.dumps(c.details, sort_keys=True, default=str)}" if c.details else ""
            lines.append(f"- {status}: {c.name} - {c.message}.{details}\n")
        lines.append("\n")
    return "".join(lines)


def _cv(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and float(v) > 0]
    if len(vals) < 2:
        return None
    m = statistics.fmean(vals)
    return round(statistics.stdev(vals) / m, 5) if m > 0 else None


def _validate_ref_speedup(val: Any) -> tuple[bool, str]:
    """reward.md gate 6.  The oracle is NEVER run at scoring; ref_speedup is a frozen constant."""
    if val is None:
        return False, ("ref_speedup is NOT calibrated in the frozen manifest. reward.md requires "
                       "reward = min(1, ln(speedup/ref_speedup)/ln(ref_speedup)); with no reference "
                       "constant the task cannot be scored -> HARD FAIL 0 (never silently 1.0)")
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False, f"ref_speedup={val!r} is not a number"
    if not math.isfinite(f) or f <= 1.0:
        return False, (f"ref_speedup={f:.6f} <= 1.0 — the reference solution does not beat the "
                       "strong baseline, so the task is not a valid performance task")
    return True, f"ref_speedup={f:.6f} (> 1.0, frozen constant read from the manifest)"


def _hard_fail_reasons(checks: dict[str, list[Check]]) -> list[str]:
    """Map failed checks onto the reward.md pre-gate vocabulary."""
    reasons: list[str] = []
    for group in checks.values():
        for c in group:
            if c.passed:
                continue
            n = c.name.lower()
            if "tamper" in n or "pristine" in n:
                reasons.append(_TAMPER_REASON)
            elif "ref_speedup" in n:
                reasons.append("ref_speedup_invalid_or_missing")
            elif "gate 5" in n or "crossed the strong baseline" in n:
                reasons.append("speedup_not_above_baseline")
            elif "identity" in n:
                reasons.append("cheating_identity_paired_ratios")
            elif "consistency" in n or "anti-cache" in n or "shape" in n:
                reasons.append("correctness_failed")
            elif "required file" in n or "loads" in n or "frozen surface" in n or "verifier error" in n:
                reasons.append("build_or_entry_contract_failed")
            else:
                reasons.append("correctness_failed")
    # stable de-dup
    seen: set[str] = set()
    return [r for r in reasons if not (r in seen or seen.add(r))]


def leaderboard_metrics(checks: dict[str, list[Check]]) -> dict[str, Any]:
    out: dict[str, Any] = {"score_metric": "log_speedup_vs_ref_speedup"}
    for group in checks.values():
        for c in group:
            for key in ("candidate_time_sec", "baseline_time_sec", "speedup", "ref_speedup",
                        "abba_pairs", "abba_pair_ratios", "cv_baseline", "cv_candidate",
                        "calibrated_baseline_time_sec_diagnostic",
                        "consistency_fraction", "matched", "mismatched"):
                if key in c.details:
                    out[key] = c.details[key]
    return out


def compute_log_reward(all_hard_pass: bool, checks: dict[str, list[Check]],
                       cfg: dict[str, Any]) -> float:
    """reward.md 性能类: reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, range [0, 1]."""
    if not all_hard_pass:
        return 0.0
    ref_ok, _ = _validate_ref_speedup(cfg.get("ref_speedup"))
    if not ref_ok:
        return 0.0
    ref_speedup = float(cfg["ref_speedup"])
    lb = leaderboard_metrics(checks)
    speedup = lb.get("speedup")
    if not isinstance(speedup, (int, float)) or not math.isfinite(float(speedup)):
        return 0.0
    speedup = float(speedup)
    if speedup <= 1.0:                                   # reward.md gate 5
        return 0.0
    if speedup < float(cfg["min_speedup_plausible"]):     # anti-noise floor
        return 0.0
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0))))


def write_outputs(checks: dict[str, list[Check]], workspace: Path, cfg: dict[str, Any]) -> bool:
    # 🔴 V7: last line of defence — nothing may have been swapped between the ABBA
    # measurement and the reward arithmetic.
    _assert_pristine("write_outputs", cfg)
    total = sum(len(group) for group in checks.values())
    passed = sum(1 for group in checks.values() for c in group if c.passed)
    all_hard_pass = all(c.passed for group in checks.values() for c in group if c.hard)
    all_pass = all(c.passed for group in checks.values() for c in group)
    lb = leaderboard_metrics(checks)
    reward = compute_log_reward(all_hard_pass, checks, cfg)
    binary_floor = 1 if all_pass else 0
    hard_fail_reasons = [] if reward > 0.0 else _hard_fail_reasons(checks)
    if reward <= 0.0 and not hard_fail_reasons:
        hard_fail_reasons = ["speedup_not_above_baseline"]
    speedup_val = lb.get("speedup") if isinstance(lb.get("speedup"), (int, float)) else None
    cv = {"baseline": lb.get("cv_baseline"), "candidate": lb.get("cv_candidate")}

    metrics = {
        "task_type": "performance",
        "reward": reward,
        "partial_score": reward,
        "binary_pass": binary_floor,
        "all_hard_gates_pass": all_hard_pass,
        "hard_fail_reasons": hard_fail_reasons,
        "speedup": speedup_val,
        "ref_speedup": cfg.get("ref_speedup"),
        "cv": cv,
        "reward_form": "reward.md 性能类: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 in [0,1]; speedup = median over >=5 ABBA pairs of baseline_time/candidate_time (baseline RE-MEASURED in-session); 0 on any consistency/anti-cache/plausibility fail, on speedup<=1, or on a missing/invalid ref_speedup",
        "strong_baseline_time_sec_diagnostic": cfg.get("strong_baseline_time_sec"),
        "passed": passed, "total": total, "pass_rate": passed / total if total else 0.0,
        "leaderboard": lb,
        "failed_checks": [
            {"category": cat, "name": c.name, "message": c.message, "hard": c.hard, "details": c.details}
            for cat, group in checks.items() for c in group if not c.passed
        ],
    }

    (workspace / "verification_report.md").write_text(markdown_report(checks), encoding="utf-8")
    vdir = Path("/logs/verifier")
    if vdir.exists():
        (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (vdir / "benchmark_results.json").write_text(json.dumps(lb, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (vdir / "correctness_results.json").write_text(json.dumps({"binary_pass": binary_floor, "all_hard_gates_pass": all_hard_pass, "passed": passed, "total": total}, indent=2) + "\n", encoding="utf-8")
        (vdir / "verifier_state.json").write_text(json.dumps({"task_type": "performance", "reward": reward, "hard_fail_reasons": hard_fail_reasons, "speedup": speedup_val, "ref_speedup": cfg.get("ref_speedup"), "passed": passed, "total": total}, indent=2, default=str) + "\n", encoding="utf-8")
        # 🔴 the authoritative result JSON shape (reward.md §结果 JSON)
        (vdir / "reward.json").write_text(json.dumps({"task_type": "performance", "reward": reward, "hard_fail_reasons": hard_fail_reasons, "speedup": speedup_val, "ref_speedup": cfg.get("ref_speedup"), "cv": cv, "binary_pass": binary_floor}, indent=2, default=str) + "\n", encoding="utf-8")
        # 🔴 reward.txt carries the FINAL NUMERIC reward (was a 1/0 binary_pass flag before
        #    2026-07-27 — that contradicted the 5-file contract and the reward spec).
        (vdir / "reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")
    adir = Path("/logs/artifacts")
    if adir.exists():
        for name in ("scoring_pipeline.py", "scoring_config.json", "verification_report.md", "action.log"):
            p = workspace / name
            if p.exists():
                try:
                    shutil.copy2(p, adir / name)
                except Exception:
                    pass
    return all_hard_pass


def main() -> int:
    # 🔴 V7: bind the guard, the suite and the writer as LOCALS of this frame BEFORE
    # any candidate code can run, so a `sys.modules["__main__"].write_outputs = …`
    # style patch is both unreachable from here AND reported by _assert.
    _assert = _assert_pristine
    _snap_cfg = _snapshot_cfg
    _run = run_all
    _write = write_outputs
    _tamper = _emit_tamper_contract
    cfg = load_manifest()
    _snap_cfg(cfg)
    workspace = WORKSPACE if WORKSPACE.exists() else Path.cwd()
    sanitize_python_path(workspace)
    try:
        checks = _run(workspace, cfg)
        _assert("post-suite, pre-reward", cfg)
    except HarnessTamper as exc:
        _tamper(str(exc))
        return 1
    except Exception as exc:
        checks = {"verifier_error": [result("Verifier error", False, f"{type(exc).__name__}: {exc}")]}
        try:
            _assert("post-verifier-error", cfg)
        except HarnessTamper as texc:
            _tamper(str(texc))
            return 1
    all_hard_pass = False
    try:
        all_hard_pass = _write(checks, workspace, cfg)
    except HarnessTamper as exc:
        _tamper(str(exc))
        return 1
    print(markdown_report(checks))
    return 0 if all_hard_pass else 1


# 🔴 V7 trust boundary: bind every measurement primitive to its ORIGINAL object.
# This runs at module import — i.e. before main() and therefore long before
# import_solution() can execute a single line of candidate code.
_snapshot_primitives()


if __name__ == "__main__":
    raise SystemExit(main())
