#!/usr/bin/env python3
"""REVIEWER-ONLY probe: which burst CONCURRENCY actually exercises the scheduling axis?

Owner ruling 2026-07-28: the fix for vllm-scheduler is a metric/instruction mismatch fix, not
headroom manufacture — RAISE the concurrency of the burst workloads until they measurably respond
to scheduling, keep >=3 responding workloads in the timed median, and move the sequential
single-request anchors out of the median (they are structurally insensitive: 0.96-1.05 in every
configuration tried) while keeping them as latency-regression gates.

Evidence that motivated it: with the rebased (--max-num-seqs 64) baseline, burst_128 measured
1.29-1.37 across 5 full graded runs while burst_64 and burst_48 sat at 0.98-1.05 — i.e. their
concurrency was simply too low to stress the scheduler on this hardware.

This probe measures ONE baseline launch and ONE candidate launch across a whole ladder of
candidate burst shapes, so the scored set can be chosen by measurement in a single pass.
Never baked, never part of tests/, never run at scoring.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-dir", default="/tests")
    ap.add_argument("--baseline", default="/tests/launch_baseline.sh")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="/app/model")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--baseline-cache", default="")
    a = ap.parse_args()

    sys.path.insert(0, a.tests_dir)
    import compute_reward as CR  # noqa: N813

    mb = CR._mixed_batch  # short_n, long_n, med_n

    # The ladder. Each entry keeps the SHAPE of one of the three shipped bursts and only scales
    # the concurrency, plus two new high-concurrency mixes.
    LADDER = [
        # short-heavy mix (shipped at 64) -> raise
        {"name": "short_heavy_128", "mixed_messages": mb(104, 12, 12), "max_tokens": 96, "concurrency": 128},
        {"name": "short_heavy_192", "mixed_messages": mb(156, 18, 18), "max_tokens": 96, "concurrency": 192},
        {"name": "short_heavy_256", "mixed_messages": mb(208, 24, 24), "max_tokens": 96, "concurrency": 256},
        # decode-heavy mix (shipped at 128, the one that RESPONDS) -> keep + raise
        {"name": "decode_heavy_128", "mixed_messages": mb(40, 24, 64), "max_tokens": 160, "concurrency": 128},
        {"name": "decode_heavy_192", "mixed_messages": mb(60, 36, 96), "max_tokens": 160, "concurrency": 192},
        {"name": "decode_heavy_256", "mixed_messages": mb(80, 48, 128), "max_tokens": 160, "concurrency": 256},
        # all-long-prefill (shipped at 48) -> raise
        {"name": "longprefill_96", "mixed_messages": mb(0, 96, 0), "max_tokens": 128, "concurrency": 96},
        {"name": "longprefill_160", "mixed_messages": mb(0, 160, 0), "max_tokens": 128, "concurrency": 160},
        # a balanced high-concurrency mix (new shape, same building blocks)
        {"name": "balanced_192", "mixed_messages": mb(64, 64, 64), "max_tokens": 128, "concurrency": 192},
    ]

    def measure(launch: str, port: int, tag: str) -> dict:
        t0 = time.time()
        with CR.server_context(launch, port, a.model_path):
            rows = CR.benchmark_server_concurrent(
                port, LADDER, warmup_iterations=a.warmup, measure_rounds=a.rounds)
        return {"tag": tag, "elapsed_s": round(time.time() - t0, 1),
                "rows": [{"name": r["name"], "median_ms": r["median_ms"], "stdev_ms": r["stdev_ms"],
                          "all_ms": r["all_ms"], "concurrency": r.get("concurrency")} for r in rows]}

    out = {"probe": "burst-concurrency ladder (reviewer-only, NOT the graded metric)",
           "rounds": a.rounds, "warmup": a.warmup}
    if a.baseline_cache and Path(a.baseline_cache).exists():
        base = json.loads(Path(a.baseline_cache).read_text())["baseline"]
        out["baseline_source"] = a.baseline_cache
    else:
        base = measure(a.baseline, CR.BASELINE_PORT, "baseline")
        out["baseline_source"] = "measured"
    out["baseline"] = base
    cand = measure(a.candidate, CR.CANDIDATE_PORT, "candidate")
    out["candidate"] = cand

    per = []
    for b, c in zip(base["rows"], cand["rows"]):
        if not b["median_ms"] or not c["median_ms"] or c["median_ms"] <= 0:
            per.append({"name": b["name"], "ratio": None, "note": "non-finite"})
            continue
        r = b["median_ms"] / c["median_ms"]
        per.append({"name": b["name"], "concurrency": b["concurrency"], "ratio": round(r, 5),
                    "baseline_ms": round(b["median_ms"], 1), "candidate_ms": round(c["median_ms"], 1),
                    "baseline_cv": (round(b["stdev_ms"] / b["median_ms"], 4) if b["median_ms"] else None),
                    "responds_ge_1_15": r >= 1.15})
    out["per_workload"] = per
    ok = [p["ratio"] for p in per if p.get("ratio")]
    out["summary"] = {"n": len(ok),
                      "n_responding_ge_1_15": sum(1 for p in per if p.get("responds_ge_1_15")),
                      "median_all": statistics.median(ok) if ok else None,
                      "max": max(ok) if ok else None, "min": min(ok) if ok else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({"summary": out["summary"], "per_workload": per}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
