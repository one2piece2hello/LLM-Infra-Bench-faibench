#!/usr/bin/env python3
"""/opt/loop/private/tests/dev_eval.py — per-round DEV proxy for e2e-vllm-scheduler-mixed-batch-serving.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh. The solver's uid cannot read it.

WHAT IT DOES (a faithful, CHEAP, leak-free mirror of the graded path, on PUBLIC prompts):
  1. Launches the candidate's /app/submission/launch_server.sh (the SAME entry the grade launches),
     waits for /health, exactly as the model-visible /app/run_dev_bench.py does.
  2. Sends a few PUBLIC prompts (the same 3 in the public dev bench) at temperature 0 and confirms
     each returns a non-empty completion — a liveness/usability correctness proxy.
  3. Measures median end-to-end latency and reports throughput = 1000 / median_ms as the RAW
     best-of-k ranking signal (higher = faster). No hidden workload, no baseline, no anchor.

WHAT IT NEVER TOUCHES / BAKES: tests/ (compute_reward.py, launch_baseline.sh, prompts.jsonl,
reward_manifest.json), the strong baseline, the calibrated ref_speedup, the hidden burst workloads.
🔴 FIDELITY NOTE: the GRADE measures an ABBA-paired MEDIAN speedup over a HIDDEN mix of single-request
latencies + concurrent heterogeneous burst makespans, AND welds a greedy token-parity gate (the
candidate must not change any greedy output). This dev proxy measures only single-request latency on
public prompts and a liveness check — it CANNOT verify token-parity against the hidden reference and
does NOT run concurrent bursts, so it is a COARSER signal than the grade: it ranks servers by
single-request speed and catches a server that fails to launch / serve, but a candidate that trades
output correctness for speed can look good on the dev proxy and still score 0 at grading. Best-of-k
still prefers faster-serving candidates; the authoritative score is the runner end-of-session grade.

REDUCED WORKLOAD (the GPU constraint): 3 public prompts x a few reps, one server launch — a per-round
pass is a single server boot + a handful of requests, vs the grade's 3 launches x 9 workload regimes.

OUTPUT: /logs/loop/dev/{verifier_state.json, reward.json}; on infra failure, harness_error.txt.
"""
from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

LOOP_PRIVATE = Path("/opt/loop/private")
MANIFEST = LOOP_PRIVATE / "manifest.json"
DEV_OUT = Path("/logs/loop/dev")
DEV_OUT.mkdir(parents=True, exist_ok=True)

INV_LAUNCH_MISSING = "launch_server_missing"
INV_SERVER_START = "server_failed_to_start"
INV_SERVE = "server_did_not_serve_public_prompts"
INV_HARNESS = "harness_error"

PUBLIC_PROMPTS = [
    ("Write a short paragraph about the ocean.", 128),
    ("List the numbers from 1 to 20, then list them again in reverse.", 128),
    ("Explain what a hash table is in two sentences.", 96),
]


def _cfg() -> dict:
    base = {"launch_server": "/app/submission/launch_server.sh", "model_path": "/app/model",
            "port": 30011, "reps": 3, "boot_timeout_sec": 1200}
    try:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        th = m.get("thresholds", m)
        for k in base:
            if k in th and th[k] is not None:
                base[k] = th[k]
    except Exception:
        pass
    return base


def _write_state(correctness_ok, failing_invariant, dev_score, dev_latency_ms, extra=None):
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {"correctness_ok": bool(correctness_ok), "hard_fail_reasons": reasons,
             "failing_invariant": failing_invariant or ""}
    if extra:
        state.update(extra)
    reward = {"dev_score": float(dev_score),
              "dev_metric": "serving throughput 1000/median_ms on PUBLIC prompts (dev proxy; higher is better; NOT your graded score)",
              "dev_latency_ms": dev_latency_ms, "correctness_ok": bool(correctness_ok)}
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, 0.0, None, {"harness_message": msg})


def _wait_health(port, proc, timeout):
    dl = time.time() + timeout
    while time.time() < dl:
        if proc.poll() is not None:
            return False
        try:
            r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-m", "4", "-w", "%{http_code}",
                                f"http://localhost:{port}/health"],
                               capture_output=True, text=True, timeout=6)
            if r.stdout.strip() == "200":
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _send(port, content, max_tokens):
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=300)
    dt = (time.perf_counter() - t) * 1000.0
    payload = json.loads(resp.read())
    text = payload["choices"][0]["message"]["content"]
    return dt, text


def main() -> int:
    cfg = _cfg()
    launch = Path(cfg["launch_server"])
    if not launch.exists():
        _write_state(False, INV_LAUNCH_MISSING, 0.0, None, {"detail": f"{launch} not found"})
        return 0
    port = int(cfg["port"])
    proc = None
    try:
        env = {**os.environ, "PORT": str(port), "MODEL_PATH": str(cfg["model_path"])}
        try:
            proc = subprocess.Popen(["bash", str(launch)], env=env, preexec_fn=os.setsid,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            _write_state(False, INV_SERVER_START, 0.0, None, {"detail": f"popen failed: {exc}"})
            return 0
        if not _wait_health(port, proc, float(cfg["boot_timeout_sec"])):
            _write_state(False, INV_SERVER_START, 0.0, None,
                         {"detail": "server never became healthy within the boot timeout"})
            return 0
        reps = int(cfg["reps"])
        lat = []
        for content, mt in PUBLIC_PROMPTS:
            for _ in range(reps):
                try:
                    dt, text = _send(port, content, mt)
                except Exception as exc:
                    _write_state(False, INV_SERVE, 0.0, None,
                                 {"detail": f"request failed: {type(exc).__name__}: {exc}"})
                    return 0
                if not (isinstance(text, str) and text.strip()):
                    _write_state(False, INV_SERVE, 0.0, None,
                                 {"detail": "server returned an empty completion (not usable)"})
                    return 0
                lat.append(dt)
        med = statistics.median(lat)
        if med <= 0:
            _write_state(False, INV_SERVE, 0.0, None, {"detail": "non-positive latency"})
            return 0
        dev_score = 1000.0 / med
        _write_state(True, None, dev_score, med,
                     {"detail": f"server healthy; served {len(lat)} public requests; "
                                f"median {med:.1f} ms -> throughput {dev_score:.3f} req/s-equiv"})
        return 0
    except Exception as exc:
        import traceback
        _harness_error(f"dev_eval crashed: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-500:]}")
        return 0
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _harness_error(f"dev_eval crashed: {type(exc).__name__}: {exc}")
        raise SystemExit(0)
