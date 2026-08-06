#!/usr/bin/env python3
"""REVIEWER-ONLY diagnostic: is the graded burst measurement CLIENT-bound?

The graded verifier drives a `concurrency`-way burst from ONE Python process with a
ThreadPoolExecutor + urllib. If that client saturates (GIL, per-request urllib
overhead), the measured makespan reflects the CLIENT and no server-side improvement
can move the ratio — which would silently cap this task's ref_speedup near 1.0.

This script measures the SAME burst against the SAME running server three ways:
  1 process  x N threads   (what the grader does)
  4 processes x N/4 threads (a client with 4x the GIL headroom)
  8 processes x N/8 threads
If the makespan falls materially as the client is parallelised, the metric is
client-bound and that is an authoring defect, not a solver-visible knob.

Never baked, never part of tests/, never run at scoring.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKER = r'''
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
port = int(sys.argv[1]); n = int(sys.argv[2]); mt = int(sys.argv[3])
slots = json.loads(sys.stdin.read())
def send(msgs):
    body = json.dumps({"model":"default","messages":msgs,"max_tokens":mt,"temperature":0}).encode()
    r = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                              headers={"Content-Type":"application/json"})
    t = time.perf_counter(); resp = urllib.request.urlopen(r, timeout=300); resp.read()
    return time.perf_counter()-t
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=max(1,n)) as pool:
    list(pool.map(send, slots[:n]))
print(json.dumps({"n": n, "wall_s": time.perf_counter()-t0}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-dir", default="/tests")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--launch", default="/tests/launch_baseline.sh")
    ap.add_argument("--model-path", default="/app/model")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.tests_dir)
    import compute_reward as CR  # noqa: N813

    Path("/tmp/clientdiag").mkdir(parents=True, exist_ok=True)
    Path("/tmp/clientdiag/w.py").write_text(WORKER)

    out: dict = {"diagnostic": "client-boundness of the graded burst measurement", "workloads": []}
    with CR.server_context(a.launch, a.port, a.model_path):
        for wl in CR.CONCURRENT_WORKLOADS:
            conc = wl.get("concurrency", 1)
            mixed = wl.get("mixed_messages") or [wl["messages"]]
            slots = [mixed[i % len(mixed)] for i in range(conc)]
            payload = json.dumps(slots)
            # warm
            for _ in range(2):
                CR.send_chat_request(a.port, slots[0], wl["max_tokens"])
            rec: dict = {"name": wl["name"], "concurrency": conc, "modes": {}}
            for nproc in (1, 4, 8):
                per = max(1, conc // nproc)
                spans = []
                for _ in range(a.rounds):
                    t0 = time.perf_counter()
                    procs = []
                    for k in range(nproc):
                        chunk = json.dumps(slots[k * per:(k + 1) * per] or slots[:per])
                        procs.append(subprocess.Popen(
                            [sys.executable, "/tmp/clientdiag/w.py", str(a.port), str(per),
                             str(wl["max_tokens"])],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
                        procs[-1].stdin.write(chunk)
                        procs[-1].stdin.close()
                    for p in procs:
                        p.wait()
                    spans.append((time.perf_counter() - t0) * 1000.0)
                rec["modes"][f"{nproc}proc_x{per}thr"] = {
                    "median_ms": round(statistics.median(spans), 2), "all_ms": [round(s, 1) for s in spans]}
            base = rec["modes"]["1proc_x%d" % conc + "thr"]["median_ms"] if f"1proc_x{conc}thr" in rec["modes"] else None
            k1 = f"1proc_x{conc}thr"
            if k1 in rec["modes"]:
                b = rec["modes"][k1]["median_ms"]
                rec["speedup_from_parallel_client"] = {
                    k: round(b / v["median_ms"], 4) for k, v in rec["modes"].items()}
            out["workloads"].append(rec)
            print(json.dumps(rec, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
