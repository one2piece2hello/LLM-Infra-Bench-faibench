#!/usr/bin/env python3
"""REVIEWER-ONLY oracle/ceiling probe for e2e-vllm-scheduler-mixed-batch-serving.

NEVER baked into the image, NEVER part of tests/, NEVER run at scoring. It exists
so the authoring lane can sweep candidate ORACLE server configurations cheaply
before paying for full graded verifier runs.

It imports the workload definitions and the measurement primitives from the FROZEN
tests/compute_reward.py so the probe measures the same shapes the grader measures,
just with fewer repetitions (declared in the output). The graded verifier has NO
fast-mode env switch — a switch there would itself be a cheat vector.

    python3 probe_pair.py --tests-dir /tests --baseline /tests/launch_baseline.sh \
        --candidate /app/submission/launch_server.sh --out /tmp/probe/x.json \
        --warmup 3 --measure 5 --burst-warmup 2 --burst-rounds 4
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
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=5)
    ap.add_argument("--burst-warmup", type=int, default=2)
    ap.add_argument("--burst-rounds", type=int, default=4)
    ap.add_argument("--skip-baseline", default="")   # path to a cached baseline json
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, a.tests_dir)
    import compute_reward as CR  # noqa: N813

    def measure(launch: str, port: int, tag: str) -> dict:
        t0 = time.time()
        with CR.server_context(launch, port, a.model_path):
            seq = CR.benchmark_server(port, CR.HIDDEN_WORKLOADS,
                                      warmup_override=a.warmup, measure_override=a.measure)
            con = CR.benchmark_server_concurrent(port, CR.CONCURRENT_WORKLOADS,
                                                 warmup_iterations=a.burst_warmup,
                                                 measure_rounds=a.burst_rounds)
        return {"tag": tag, "launch": launch, "elapsed_s": round(time.time() - t0, 1),
                "sequential": [{"name": r["name"], "median_ms": r["median_ms"],
                                "stdev_ms": r["stdev_ms"], "all_ms": r["all_ms"]} for r in seq],
                "concurrent": [{"name": r["name"], "median_ms": r["median_ms"],
                                "stdev_ms": r["stdev_ms"], "all_ms": r["all_ms"],
                                "concurrency": r.get("concurrency")} for r in con]}

    out = {"probe": "reviewer-only ceiling probe (NOT the graded metric)",
           "iterations": {"seq_warmup": a.warmup, "seq_measure": a.measure,
                          "burst_warmup": a.burst_warmup, "burst_rounds": a.burst_rounds}}

    if a.skip_baseline and Path(a.skip_baseline).exists():
        base = json.loads(Path(a.skip_baseline).read_text())
        base = base.get("baseline", base)
        out["baseline_source"] = a.skip_baseline
    else:
        base = measure(a.baseline, CR.BASELINE_PORT, "baseline")
        out["baseline_source"] = "measured"
    out["baseline"] = base
    if a.baseline_only:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(json.dumps({"baseline_only": True, "elapsed_s": base.get("elapsed_s")}))
        return 0

    cand = measure(a.candidate, CR.CANDIDATE_PORT, "candidate")
    out["candidate"] = cand

    ratios = []
    per = []
    for b, c in zip(base["sequential"] + base["concurrent"],
                    cand["sequential"] + cand["concurrent"]):
        if not b["median_ms"] or not c["median_ms"]:
            continue
        r = b["median_ms"] / c["median_ms"]
        ratios.append(r)
        per.append({"name": b["name"], "ratio": round(r, 5),
                    "baseline_ms": round(b["median_ms"], 2), "candidate_ms": round(c["median_ms"], 2)})
    seq_r = ratios[:len(base["sequential"])]
    burst_r = ratios[len(base["sequential"]):]
    out["per_workload"] = per
    out["stats"] = {
        "n_pairs": len(ratios),
        "median_all": statistics.median(ratios) if ratios else None,
        "geomean_all": (statistics.geometric_mean(ratios) if ratios else None),
        "median_sequential_only": statistics.median(seq_r) if seq_r else None,
        "median_bursts_only": statistics.median(burst_r) if burst_r else None,
        "min": min(ratios) if ratios else None, "max": max(ratios) if ratios else None,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({"stats": out["stats"], "per_workload": per}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
