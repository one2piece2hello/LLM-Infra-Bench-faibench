#!/usr/bin/env python3
"""/opt/loop/private/dev_eval.py — the per-round DEV proxy for e2e-h3-eval-harness-throughput-quality.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh (0700). The solver's uid cannot read it.

WHAT IT DOES (a faithful, leak-free mirror of the graded path, on the PUBLIC dev split):
  1. Loads the candidate's load_scoring_pipeline_for_verification from /app/submission/scoring_pipeline.py.
  2. Loads the PUBLIC, DISJOINT dev split at /data/eval_harness/dev_samples.jsonl (the "progress
     monitor" the task statement already hands the solver).
  3. Recomputes the reference per-sample scores with an INDEPENDENT implementation of the SAME
     reference semantics the task statement discloses verbatim (normalise / regex extraction /
     take_first / majority_vote / loglikelihood argmax / exact_match|contains|prefix_match), and runs
     the REAL welded EXACT-MATCH consistency gate + an anti-cache probe — the SAME correctness gate
     the grade enforces. correctness_ok reflects that gate exactly.
  4. Times the candidate's .score() in ABBA-alternating pairs against the PUBLIC naive template
     (dev_baseline_naive.py, the correct-but-slow starter that already ships in this image), and
     reports the RAW speedup = median(naive_time / candidate_time) as the best-of-k ranking signal.

WHAT IT NEVER TOUCHES / BAKES: the strong-baseline reference implementation, the held-out sample
set, the calibrated ref_speedup / strong_baseline_time_sec, the frozen manifest, or /tests. The dev
signal is a RAW speedup vs the PUBLIC naive template, never normalized against the graded anchor and
never divided by the hidden strong baseline. 🔴 FIDELITY NOTE: the graded speedup divides by the
HIDDEN strong baseline; this dev speedup divides by the PUBLIC naive template, so the numbers differ
in scale — but both are strictly monotonic in candidate throughput, so best-of-k ranking (pick the
fastest correctness-passing candidate) is faithful. The authoritative score is the runner's
end-of-session /tests grade (fresh held-out set + strong-baseline ABBA + calibrated anchor).

OUTPUT: /logs/loop/dev/{verifier_state.json, reward.json}; on infra failure, harness_error.txt.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

LOOP_PRIVATE = Path("/opt/loop/private")
MANIFEST = LOOP_PRIVATE / "manifest.json"
DEV_TESTS = LOOP_PRIVATE / "tests"
DEV_OUT = Path("/logs/loop/dev")
DEV_OUT.mkdir(parents=True, exist_ok=True)

# ---- named DEV failing-invariants (leak-free; mirror the graded gate vocabulary, dev-scoped) ----
INV_SUBMISSION_MISSING = "submission_missing"
INV_LOAD = "load_scoring_pipeline_for_verification_failed"
INV_SHAPE = "score_output_shape_invalid"
INV_CONSISTENCY = "per_sample_consistency_failed"
INV_ANTICACHE = "anti_cache_probe_failed"
INV_TIMING = "timing_invalid"
INV_HARNESS = "harness_error"

# defaults (overridable from the dev manifest; NO calibrated anchor here)
_DEFAULTS = {
    "score_atol": 1e-9,
    "abba_pairs": 6,
    "timing_repeats": 3,
    "max_score_time_sec": 900.0,
    "anti_cache_perturb_frac": 0.25,
    "probe_seed": 20240725,
    "max_samples": 200000,
    "dev_data_path": "/data/eval_harness/dev_samples.jsonl",
    "naive_baseline_impl": str(DEV_TESTS / "dev_baseline_naive.py"),
    "submission_path": "/app/submission/scoring_pipeline.py",
}


def _cfg() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        th = m.get("thresholds", m)
        for k in _DEFAULTS:
            if k in th and th[k] is not None:
                cfg[k] = th[k]
    except Exception:
        pass
    return cfg


def _write_state(correctness_ok: bool, failing_invariant, dev_score: float,
                 dev_speedup, extra: dict | None = None) -> None:
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {
        "correctness_ok": bool(correctness_ok),
        "hard_fail_reasons": reasons,
        "failing_invariant": failing_invariant or "",
    }
    if extra:
        state.update(extra)
    reward = {
        "dev_score": float(dev_score),
        "dev_metric": "speedup vs the PUBLIC naive template (dev proxy; higher is better; NOT your graded score)",
        "dev_speedup": dev_speedup,
        "correctness_ok": bool(correctness_ok),
    }
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, 0.0, None, {"harness_message": msg})


# -----------------------------------------------------------------------------
# INDEPENDENT reference scorer — the SAME semantics the task statement discloses verbatim.
# (public: the instruction + the shipped template define exactly this; nothing secret.)
# -----------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _normalise(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def _apply_filter(raw, filt: str, pattern) -> str:
    cands = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
    if pattern:
        rx = re.compile(pattern)
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


def _reference_score_one(sample: dict) -> float:
    metric = str(sample.get("metric", "exact_match"))
    if metric == "loglikelihood_acc":
        lls = [float(x) for x in sample.get("choice_loglikelihoods", [])]
        if not lls:
            return 0.0
        pred = max(range(len(lls)), key=lambda i: (lls[i], -i))
        return 1.0 if pred == int(sample.get("gold_index", -1)) else 0.0
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


def _reference_scores(samples: list) -> dict:
    return {str(s["id"]): _reference_score_one(s) for s in samples}


def _perturb_for_anticache(samples: list, frac: float, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    n = len(samples)
    idx = set(rng.sample(range(n), max(1, int(n * frac)))) if n else set()
    for i, s in enumerate(samples):
        t = dict(s)
        t["id"] = f"probe::{s['id']}"
        if i in idx:
            if t.get("metric") == "loglikelihood_acc" and t.get("choice_loglikelihoods"):
                t["choice_loglikelihoods"] = list(reversed([float(x) for x in t["choice_loglikelihoods"]]))
            else:
                r = t.get("response")
                t["response"] = [str(x)[::-1] for x in r] if isinstance(r, list) else str(r)[::-1]
        out.append(t)
    return out


def _read_jsonl(path: Path, cap: int) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("# provenance-marker"):
            continue
        rows.append(json.loads(line))
        if len(rows) >= cap:
            break
    return rows


def _import_pipeline(path: Path, mod_name: str, device: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    workspace = str(path.parent.resolve())
    inserted = workspace not in sys.path
    if inserted:
        sys.path.insert(0, workspace)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(workspace)
            except ValueError:
                pass
    loader = getattr(module, "load_scoring_pipeline_for_verification", None)
    if loader is None:
        raise RuntimeError("submission must define load_scoring_pipeline_for_verification(device)")
    sig = inspect.signature(loader)
    kwargs, positional = {}, []
    for name, par in sig.parameters.items():
        if name == "device":
            kwargs[name] = device
        elif par.default is inspect._empty and not positional:
            positional.append(device)
    pipe = loader(*positional, **kwargs)
    if pipe is None:
        raise RuntimeError("load_scoring_pipeline_for_verification returned None")
    return pipe


def _as_score_map(out):
    if not isinstance(out, (list, tuple)):
        return None
    m = {}
    for row in out:
        if not isinstance(row, dict) or "id" not in row or "score" not in row:
            return None
        try:
            m[str(row["id"])] = float(row["score"])
        except Exception:
            return None
    return m


def _consistency(cand: dict, ref: dict, atol: float):
    matched = mismatched = missing = 0
    for rid, rv in ref.items():
        if rid not in cand:
            missing += 1
        elif math.isfinite(cand[rid]) and abs(cand[rid] - rv) <= atol:
            matched += 1
        else:
            mismatched += 1
    return matched, mismatched, missing


def _time_scoring(pipe, samples, repeats: int, max_sec: float):
    times = []
    last = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = pipe.score([dict(s) for s in samples])
        dt = time.perf_counter() - t0
        if dt > max_sec:
            return (list(out) if isinstance(out, list) else []), dt, False
        times.append(dt)
        last = list(out) if isinstance(out, list) else []
    times.sort()
    return last, times[len(times) // 2], True


def main() -> int:
    cfg = _cfg()
    atol = float(cfg["score_atol"])
    sub_path = Path(cfg["submission_path"])
    if not sub_path.exists():
        _write_state(False, INV_SUBMISSION_MISSING, 0.0, None, {"detail": f"{sub_path} not found"})
        return 0

    dev_path = Path(cfg["dev_data_path"])
    if not dev_path.exists():
        _harness_error(f"public dev split not found at {dev_path}")
        return 0
    try:
        samples = _read_jsonl(dev_path, int(cfg["max_samples"]))
    except Exception as exc:
        _harness_error(f"could not read dev split: {type(exc).__name__}: {exc}")
        return 0
    if not samples:
        _harness_error("public dev split is empty")
        return 0

    device = "cpu"
    # --- load candidate ---
    try:
        cand_pipe = _import_pipeline(sub_path, "dev_submitted_scorer", device)
    except Exception as exc:
        # an import/traceback here is a candidate defect (bad code), not infra -> correctness FAIL.
        _write_state(False, INV_LOAD, 0.0, None, {"detail": f"{type(exc).__name__}: {exc}"})
        return 0
    if not (hasattr(cand_pipe, "score") and callable(getattr(cand_pipe, "score"))):
        _write_state(False, INV_LOAD, 0.0, None, {"detail": "pipeline has no callable .score"})
        return 0

    ref = _reference_scores(samples)
    max_sec = float(cfg["max_score_time_sec"])
    repeats = int(cfg["timing_repeats"])

    # --- timed correctness pass (real welded consistency gate) ---
    try:
        cand_out, cand_time, ok = _time_scoring(cand_pipe, samples, repeats, max_sec)
    except Exception as exc:
        _write_state(False, INV_TIMING, 0.0, None, {"detail": f"score() raised: {type(exc).__name__}: {exc}"})
        return 0
    if not ok:
        _write_state(False, INV_TIMING, 0.0, None, {"detail": f"scoring pass exceeded {max_sec:.0f}s cap"})
        return 0
    cand_map = _as_score_map(cand_out)
    if cand_map is None:
        _write_state(False, INV_SHAPE, 0.0, None, {"detail": "expected list of {'id','score'} rows"})
        return 0
    matched, mismatched, missing = _consistency(cand_map, ref, atol)
    if mismatched != 0 or missing != 0:
        _write_state(False, INV_CONSISTENCY, 0.0, None,
                     {"detail": f"matched={matched}/{len(ref)} mismatched={mismatched} missing={missing}"})
        return 0

    # --- anti-cache probe (perturbed inputs, fresh ids) ---
    probe = _perturb_for_anticache(samples, float(cfg["anti_cache_perturb_frac"]), int(cfg["probe_seed"]))
    probe_ref = _reference_scores(probe)
    try:
        probe_out = cand_pipe.score([dict(s) for s in probe])
    except Exception as exc:
        _write_state(False, INV_ANTICACHE, 0.0, None, {"detail": f"probe score() raised: {type(exc).__name__}: {exc}"})
        return 0
    probe_map = _as_score_map(probe_out)
    if probe_map is None:
        _write_state(False, INV_ANTICACHE, 0.0, None, {"detail": "probe .score output malformed"})
        return 0
    p_match, p_mis, p_miss = _consistency(probe_map, probe_ref, atol)
    if p_mis != 0 or p_miss != 0:
        _write_state(False, INV_ANTICACHE, 0.0, None,
                     {"detail": f"probe matched={p_match}/{len(probe_ref)} mismatched={p_mis} missing={p_miss}"})
        return 0

    # --- ABBA timing vs the PUBLIC naive template (raw monotonic ranking signal) ---
    naive_path = Path(cfg["naive_baseline_impl"])
    dev_speedup = None
    if naive_path.exists():
        try:
            naive_pipe = _import_pipeline(naive_path, "dev_naive_baseline", device)
            n_pairs = max(4, int(cfg["abba_pairs"]))
            ratios = []
            ok_pairs = True
            for i in range(n_pairs):
                if i % 2 == 0:
                    _, bt, bok = _time_scoring(naive_pipe, samples, 1, max_sec)
                    _, ct, cok = _time_scoring(cand_pipe, samples, 1, max_sec)
                else:
                    _, ct, cok = _time_scoring(cand_pipe, samples, 1, max_sec)
                    _, bt, bok = _time_scoring(naive_pipe, samples, 1, max_sec)
                if not (bok and cok) or bt <= 0 or ct <= 0:
                    ok_pairs = False
                    break
                ratios.append(bt / ct)
            if ok_pairs and ratios:
                dev_speedup = float(statistics.median(ratios))
        except Exception:
            dev_speedup = None

    # correctness passed. dev_score = raw speedup vs naive (monotonic in candidate speed); if the
    # ABBA measurement could not be produced, fall back to a correctness-only positive signal so a
    # correct candidate still ranks above the (non-passing) baseline.
    if isinstance(dev_speedup, (int, float)) and math.isfinite(dev_speedup) and dev_speedup > 0:
        dev_score = dev_speedup
    else:
        dev_score = 1.0
    _write_state(True, None, dev_score, dev_speedup,
                 {"detail": f"consistency matched={matched}/{len(ref)}; dev_speedup_vs_naive={dev_speedup}"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _harness_error(f"dev_eval crashed: {type(exc).__name__}: {exc}")
        raise SystemExit(0)
