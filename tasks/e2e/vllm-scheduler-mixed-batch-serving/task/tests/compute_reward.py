"""Correctness-gated verifier for the vLLM request-scheduling / continuous-batching
serving-optimization task.

Reward — reward.md 性能类, BOUNDED [0,1] (2026-07-27 rewrite; the previous shape was an
unbounded geometric-mean speedup in which tying the baseline was already full marks):

    speedup = median over the ABBA-paired per-workload ratios (baseline_ms / candidate_ms)
    reward  = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0          in [0, 1]

  * speedup == ref_speedup -> 0.0 (must EXCEED it to score)
  * speedup >= ref_speedup**2   -> 1.0 (cap)
  * speedup <= 1.0              -> 0   (tying the baseline is worth NOTHING now)

`ref_speedup` is an AUTHORING-TIME FROZEN CONSTANT read from tests/reward_manifest.json,
which is uploaded FRESH with tests/ at scoring. The oracle is NOT in the image and is
NEVER run here. A missing/invalid constant is a NAMED hard fail, never a silent 1.0.

ABBA pairing: Phase 1 baseline server -> Phase 2 candidate server -> Phase 3 baseline
re-check (B A B). Each of the 6 hidden workloads contributes ONE paired ratio, whose
denominator is the MEAN of that workload's two baseline measurements; the per-workload
figure inside each phase is itself a median over the in-phase repetitions. 6 pairs >= the
5 pairs reward.md requires. (A full 5x alternation of the WHOLE suite is not affordable:
every phase is a separate vLLM server launch.)

reward.md 性能类 pre-gates — any one hit => reward 0 with a NAMED reason:
  1 build_or_entry_contract_failed  (launch script / model / server bring-up / readiness)
  2 correctness_failed              (token-parity gate below threshold, any case)
  3 cheating_detected               (frozen-surface reference, timer tamper, paired ratios
                                     pinned at 1.0, decode-work probe non-response)
  4 forbidden_edit_path             (candidate touched the frozen evaluation surface)
  5 speedup_not_above_baseline      (speedup <= 1)
  6 ref_speedup_invalid_or_missing  (no calibrated constant, or <= 1)

Flow:
1. Launch baseline server -> benchmark speed (sequential + concurrent) + reference outputs
2. Launch candidate server -> outputs, token comparison, decode-work probe, benchmark speed
3. Baseline re-check (the B of B-A-B; sequential AND concurrent)
4. Paired-median speedup -> bounded log reward
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable or "python3"
REWARD_MANIFEST = SCRIPT_DIR / "reward_manifest.json"
TASK_ID = "e2e-vllm-scheduler-mixed-batch-serving"

# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------
BASELINE_PORT = 30000
CANDIDATE_PORT = 30001
SERVER_STARTUP_TIMEOUT = 1800  # seconds. Heavy configs (spec-v2 + fp8 + deep_gemm + fused-qk-norm-rope + cutedsl) can take 10-20 min for CUDA graphs + FlashInfer JIT + first-generation warmup. 900s proved too tight in practice.
REQUEST_TIMEOUT = 300  # seconds per request (warmup requests can be slow)

# ---------------------------------------------------------------------------
# Benchmark parameters.  Heavy warmup to cover CUDA graph compilation,
# FlashInfer autotuning, KV cache page allocation, and torch JIT.
# ---------------------------------------------------------------------------
WARMUP_ITERATIONS = 20
MEASURE_ITERATIONS = 20
RECHECK_WARMUP = 10
RECHECK_ITERATIONS = 10
# The third measurement phase now also re-measures the CONCURRENT bursts, so all 6 workloads are
# real ABBA pairs (it used to re-check only the 3 sequential anchors).
RECHECK_CONC_WARMUP = 5
RECHECK_CONC_ROUNDS = 6
# The sequential single-request cases are no longer in the timed median (they are structurally
# scheduling-insensitive: 0.96-1.05 in every configuration measured). They are a LATENCY-
# REGRESSION gate: the candidate must not make single-request latency materially worse while
# optimising the bursts. Tolerance calibrated from measurement — the WORST honest sequential
# ratio observed across 13 oracle configurations and 5 full graded runs was 0.9579, so this
# tolerance sits well below anything an honest submission produced.
SEQ_REGRESSION_MIN_RATIO = 0.85
SEQ_GATE_WARMUP = 5
SEQ_GATE_MEASURE = 8
VARIANCE_THRESHOLD = 0.30  # flag if baseline re-check differs >30%
MIN_PAIRS = 5              # reward.md: median over >= 5 ABBA pairs
PINNED_RATIO_TOL = 1e-4    # reward.md gate 3: [1.0, 1.0, ...] identical paired ratios
DECODE_PROBE_MIN_RATIO = 1.5  # reward.md gate 3: the sleep-probe analogue must respond

# ---------------------------------------------------------------------------
# Token-level correctness gate.
# Runs 1000+ prompts on both baseline and candidate, compares outputs
# token-by-token (whitespace-split).  The candidate must match the baseline's
# greedy outputs within the threshold to receive a score.
# ---------------------------------------------------------------------------
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.jsonl"
TOKEN_MATCH_THRESHOLD = 0.95  # average per-prompt token match rate required
MIN_VALID_OUTPUTS = 500  # minimum baseline outputs needed for a valid comparison
CONSECUTIVE_FAILURE_LIMIT = 50  # abort output collection after this many in a row

# ---------------------------------------------------------------------------
# Mixed-workload building blocks. This task's HIDDEN workloads deliberately live
# in the SCHEDULER-BOUND regime: a *mix* of short and long prefills contending
# with ongoing decode, at concurrency levels high enough that the continuous-
# batching scheduler's admission / batching / chunking / preemption decisions
# — not raw kernel speed — determine end-to-end latency. This corner of the
# workload space is where request scheduling moves the metric, and it is
# disjoint from the public dev bench.
# ---------------------------------------------------------------------------
_PARA_A = (
    "The development of artificial intelligence has been one of the most "
    "transformative technological advances of the modern era. Beginning with "
    "Alan Turing's 1950 paper 'Computing Machinery and Intelligence', which "
    "proposed the Turing test, the field evolved through symbolic AI, the "
    "connectionist revival of the 1980s, the 2012 AlexNet breakthrough, and the "
    "2017 transformer architecture of Vaswani et al. Large language models then "
    "demonstrated broad capabilities across natural language tasks. "
)
_PARA_B = (
    "The Industrial Revolution began in Britain in the late eighteenth century, "
    "starting with mechanised textile manufacturing and expanding to steam power, "
    "iron production, and railways. Its second phase brought electricity, the "
    "internal-combustion engine, and mass production, driving urbanisation, labour "
    "movements, and eventually the large-scale burning of fossil fuels. "
)


def _ctx(n_reps: int) -> str:
    blocks = [_PARA_A, _PARA_B]
    return "".join(blocks[i % 2] for i in range(n_reps))


# A long prefill request — a big chunk of prefill work the scheduler must
# interleave with (not stall) ongoing decode.
_LONG_PREFILL = (
    f"Read the following report carefully:\n\n{_ctx(40)}\n\n"
    "Write a detailed multi-paragraph analysis of its themes."
)
# A short prefill request — cheap prefill, mostly decode; latency-sensitive.
_SHORT_PROMPT = "In one sentence, what is the capital of France?"
_MED_PROMPT = (
    "Explain the difference between TCP and UDP protocols. "
    "Cover reliability, ordering, and common use cases."
)

# ---------------------------------------------------------------------------
# Hidden sequential workloads — the per-shape latency anchors.
# ---------------------------------------------------------------------------
HIDDEN_WORKLOADS = [
    # Short in / short out — the latency-sensitive interactive request that a
    # naive scheduler starves behind long prefills.
    {
        "name": "short_in_short_out",
        "messages": [{"role": "user", "content": _SHORT_PROMPT}],
        "max_tokens": 48,
    },
    # Long prefill / medium out — the heavy prefill unit that a chunked-prefill /
    # admission policy must schedule without blocking decode.
    {
        "name": "longprefill_med_out",
        "messages": [{"role": "user", "content": _LONG_PREFILL}],
        "max_tokens": 128,
    },
    # Medium in / long out — a sustained decode stream.
    {
        "name": "med_in_long_out",
        "messages": [{"role": "user", "content": _MED_PROMPT}],
        "max_tokens": 384,
    },
]

# ---------------------------------------------------------------------------
# Concurrent MIXED workloads — the heart of this task. Each round dispatches a
# HETEROGENEOUS batch (short + long-prefill + medium requests together) so the
# scheduler faces the real problem: admit and batch a burst of dissimilar
# requests without head-of-line blocking, KV exhaustion, or preemption thrash.
# `mixed_messages` gives each of the `concurrency` slots a distinct request; the
# per-request latency across the whole heterogeneous batch is what is measured.
# ---------------------------------------------------------------------------


def _mixed_batch(short_n: int, long_n: int, med_n: int) -> list:
    slots: list = []
    for _ in range(short_n):
        slots.append([{"role": "user", "content": _SHORT_PROMPT}])
    for _ in range(long_n):
        slots.append([{"role": "user", "content": _LONG_PREFILL}])
    for _ in range(med_n):
        slots.append([{"role": "user", "content": _MED_PROMPT}])
    return slots


CONCURRENT_WORKLOADS = [
    # 🔴 RE-SCOPED 2026-07-28 (owner ruling). The shipped set was 64/128/48-way; MEASURED across
    # 5 full graded runs, only the 128-way burst responded to scheduling (1.29-1.37) while the
    # 64- and 48-way bursts sat at 0.98-1.05 — their concurrency was simply too low to stress the
    # scheduler on this hardware. The SHAPES are unchanged (same short / long-prefill / medium
    # building blocks); only the concurrency is raised, plus one balanced high-concurrency mix.
    # Selection was by measurement: see tests/reward_manifest.json -> rescope_2026_07_28.
    # These are the TIMED workloads. The sequential single-request cases below are NOT timed any
    # more — an insensitive workload inside the median is noise with a vote — they are a latency-
    # REGRESSION gate instead, so a candidate cannot win the bursts by tanking single-request
    # latency.
    {
        "name": "burst_256_short_heavy_mix",
        "mixed_messages": _mixed_batch(short_n=208, long_n=24, med_n=24),
        "max_tokens": 96,
        "concurrency": 256,
        "measured_oracle_ratio": 1.13884,
    },
    {
        "name": "burst_128_decode_heavy_mix",
        "mixed_messages": _mixed_batch(short_n=40, long_n=24, med_n=64),
        "max_tokens": 160,
        "concurrency": 128,
        "measured_oracle_ratio": 1.31012,
    },
    {
        "name": "burst_192_decode_heavy_mix",
        "mixed_messages": _mixed_batch(short_n=60, long_n=36, med_n=96),
        "max_tokens": 160,
        "concurrency": 192,
        "measured_oracle_ratio": 1.23489,
    },
    {
        "name": "burst_256_decode_heavy_mix",
        "mixed_messages": _mixed_batch(short_n=80, long_n=48, med_n=128),
        "max_tokens": 160,
        "concurrency": 256,
        "measured_oracle_ratio": 1.22198,
    },
    {
        "name": "burst_96_all_longprefill",
        "mixed_messages": _mixed_batch(short_n=0, long_n=96, med_n=0),
        "max_tokens": 128,
        "concurrency": 96,
        "measured_oracle_ratio": 1.50889,
    },
    {
        "name": "burst_160_all_longprefill",
        "mixed_messages": _mixed_batch(short_n=0, long_n=160, med_n=0),
        "max_tokens": 128,
        "concurrency": 160,
        "measured_oracle_ratio": 1.30846,
    },
]
# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--app-dir", default="/app")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-time-ms", type=int, default=0)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--fail", type=str, default=None)
    p.add_argument("--fail-gate", type=str, default="build_or_entry_contract_failed",
                   help="the NAMED reward.md pre-gate that --fail maps to")
    return p.parse_args()


# ===================================================================
# Frozen reward manifest (uploaded fresh with tests/; the oracle is never run)
# ===================================================================

def load_reward_manifest() -> dict:
    try:
        return json.loads(REWARD_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def manifest_ref_speedup(cfg: dict) -> float | None:
    r = cfg.get("ref_speedup")
    if isinstance(r, dict):
        r = r.get("value")
    try:
        r = None if r is None else float(r)
    except (TypeError, ValueError):
        return None
    if r is None or not math.isfinite(r):
        return None
    return r


def bounded_log_reward(speedup: float | None, ref: float | None) -> tuple[float, list[str]]:
    """reward.md 性能类: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, range [0,1]."""
    if ref is None or not math.isfinite(ref) or ref <= 1.0:
        return 0.0, ["ref_speedup_invalid_or_missing"]
    if speedup is None or not math.isfinite(speedup) or speedup <= 1.0:
        return 0.0, ["speedup_not_above_baseline"]
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0)))), []


# ===================================================================
# Reward output — the FULL 5-file /logs/verifier contract on EVERY path
# ===================================================================

def emit_reward(
    output_dir: str,
    score: float,
    reason: str,
    total_time_ms: int,
    subscores: list[dict] | None = None,
    additional_data: dict | None = None,
    *,
    hard_fail_reasons: list[str] | None = None,
    speedup: float | None = None,
    ref_speedup: float | None = None,
    cv: dict | None = None,
    correctness: dict | None = None,
    benchmark: dict | None = None,
) -> None:
    """Write reward.json + reward.txt + metrics.json + verifier_state.json +
    correctness_results.json + benchmark_results.json.

    🔴 Every hard-fail short-circuit goes through HERE, so a failing path can never emit
    a partial contract (an earlier shape wrote 2 files, and reward 0 with an empty
    hard_fail_reasons list). `total` is always an int, never null.
    """
    score = float(score or 0.0)
    hard = list(hard_fail_reasons or [])
    if score <= 0.0 and not hard:
        hard = ["zero_without_named_reason"]
    corr = correctness or {}
    total = int(corr.get("compared") or corr.get("total_prompts") or 0)
    passed = int(corr.get("exact_matches") or 0)
    payload = {
        "task_type": "performance",
        "score": score,
        "reward": score,
        "hard_fail_reasons": hard,
        "speedup": speedup,
        "ref_speedup": ref_speedup,
        "cv": cv or {"baseline": 0.0, "candidate": 0.0},
        "metric_kind": "wallclock_latency_ratio",
        "metric_name": "median_paired_speedup_over_the_concurrent_burst_workloads",
        "metric_direction": "higher_is_better",
        "timing_measured": True,
        "reward_form": ("reward.md 性能类: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 in [0,1]; "
                        "speedup = MEDIAN of the ABBA-paired per-burst ratios "
                        "(baseline_ms/candidate_ms, B-A-B, the sequential and concurrent "
                        "denominators are the mean of the two baseline phases); ref_speedup is "
                        "a FROZEN authoring-time constant from tests/reward_manifest.json. "
                        "speedup <= 1 (a candidate that only ties the baseline) => 0."),
        "subscores": subscores or [],
        "additional_data": {
            **(additional_data or {}),
            "reason": reason,
            "total_time_ms": total_time_ms,
        },
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        (out_dir / "reward.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (out_dir / "reward.txt").write_text(f"{score}\n")
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (out_dir / "verifier_state.json").write_text(json.dumps({
            "task_id": TASK_ID, "task_type": "performance", "reward": score,
            "hard_fail_reasons": hard, "speedup": speedup, "ref_speedup": ref_speedup,
            "correctness_ok": bool(score > 0.0 or (corr.get("token_match_rate") or 0)
                                   >= TOKEN_MATCH_THRESHOLD),
            "passed": passed, "total": total,
        }, indent=2, default=str) + "\n")
        (out_dir / "correctness_results.json").write_text(json.dumps({
            "token_match_rate": corr.get("token_match_rate"),
            "exact_match_rate": corr.get("exact_match_rate"),
            "threshold": TOKEN_MATCH_THRESHOLD,
            "passed": passed, "total": total,
            "compared": corr.get("compared"), "total_prompts": corr.get("total_prompts"),
            "mismatches": corr.get("mismatches") or [],
        }, indent=2, default=str) + "\n")
        (out_dir / "benchmark_results.json").write_text(json.dumps(
            benchmark or {"speedup": speedup, "ref_speedup": ref_speedup,
                          "paired_ratios": [], "note": reason},
            indent=2, default=str) + "\n")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    print(json.dumps({k: payload[k] for k in
                      ("reward", "hard_fail_reasons", "speedup", "ref_speedup")},
                     indent=2, default=str))


# ===================================================================
# Server lifecycle
# ===================================================================

# ---------------------------------------------------------------------------
# Diagnostic logging — writes to /logs/verifier/ alongside the server output
# so we can debug stalls without the pipe buffer held by Popen.
# ---------------------------------------------------------------------------
VERIFIER_LOG_DIR = os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier")

def _diag_log(tag: str, msg: str) -> None:
    """Append a timestamped line to /logs/verifier/diag.log (best-effort)."""
    try:
        os.makedirs(VERIFIER_LOG_DIR, exist_ok=True)
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{tag}] {msg}\n"
        with open(os.path.join(VERIFIER_LOG_DIR, "diag.log"), "a") as f:
            f.write(line)
    except Exception:
        pass
    # Also echo to stdout for live Modal exec stream
    print(f"[diag] [{tag}] {msg}", flush=True)


def _dump_server_state(tag: str, port: int) -> None:
    """When a timeout fires, capture as much state as possible for later triage."""
    try:
        os.makedirs(VERIFIER_LOG_DIR, exist_ok=True)
        dump_path = os.path.join(VERIFIER_LOG_DIR, f"diag_dump_{tag}.txt")
        parts = [f"=== DIAG DUMP ({tag}) port={port} at {time.strftime('%Y-%m-%d %H:%M:%S')} ==="]
        def shell(cmd):
            try:
                r = subprocess.run(["bash","-c",cmd], capture_output=True, text=True, timeout=10)
                return f"$ {cmd}\n{r.stdout}{r.stderr}"
            except Exception as e:
                return f"$ {cmd} (err: {e})"
        parts += [
            shell("date"),
            shell("ps -eo pid,ppid,etime,stat,command | grep -E 'sglang|launch_server|compute_reward' | grep -v grep | head -20"),
            shell("awk '$4==\"0A\" {print $0}' /proc/net/tcp"),
            shell("awk '$4==\"0A\" {print $0}' /proc/net/tcp6"),
            shell(f"curl -sS -m 3 -w 'HTTP %{{http_code}}\\n' -o /dev/null http://localhost:{port}/v1/models 2>&1"),
            shell(f"curl -sS -m 3 -w 'HTTP %{{http_code}}\\n' -o /dev/null http://localhost:{port}/health 2>&1"),
            shell("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"),
            shell("free -g | head -3"),
        ]
        with open(dump_path, "w") as f:
            f.write("\n".join(parts) + "\n")
        _diag_log(tag, f"state dump written to {dump_path}")
    except Exception as e:
        _diag_log(tag, f"dump failed: {e}")


def wait_for_server(
    port: int, timeout: int = SERVER_STARTUP_TIMEOUT, proc: subprocess.Popen | None = None,
) -> None:
    # Three-stage readiness to work around SGLang warmup behavior on heavy
    # configs (spec + fp8 + deep_gemm): the /health endpoint does an internal
    # generation and returns 503 until ServerStatus flips to Up, which can take
    # 5-15+ minutes after the socket binds.
    #
    # Stage 1: TCP connect — raw socket probe, fastest signal that uvicorn is up.
    # Stage 2: GET /v1/models — confirms HTTP handlers are mounted.
    # Stage 3: POST /v1/chat/completions (max_tokens=1) — confirms scheduler can
    #          actually generate. Uses curl with hard -m timeout so a stuck
    #          socket can't pin the whole budget like urllib can.
    import socket
    t0 = time.time()
    deadline = t0 + timeout
    last_err = ""
    _diag_log(f"wait_port_{port}", f"begin; budget={timeout}s")

    def subprocess_died():
        if proc is not None and proc.poll() is not None:
            # proc.stdout is now a file (see server_context). Read the tail from disk.
            stdout = ""
            try:
                log_path = os.path.join(VERIFIER_LOG_DIR, f"server_{port}.log")
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        data = f.read()
                        stdout = data[-2000:]
            except Exception:
                pass
            _diag_log(f"wait_port_{port}", f"subprocess died rc={proc.returncode} tail={stdout[-500:]}")
            raise RuntimeError(
                f"Server process exited with code {proc.returncode} "
                f"before becoming ready.\nLast output:\n{stdout}"
            )

    # Stage 1: TCP bind (should be fast).
    probes = 0
    while time.time() < deadline:
        subprocess_died()
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                elapsed = time.time() - t0
                _diag_log(f"wait_port_{port}", f"stage1 TCP bound at t={elapsed:.1f}s ({probes+1} probes)")
                break
        except (OSError, socket.timeout) as e:
            last_err = f"TCP: {e}"
        probes += 1
        if probes % 30 == 0:
            _diag_log(f"wait_port_{port}", f"stage1 still trying t={int(time.time()-t0)}s last={last_err}")
        time.sleep(2)
    else:
        _dump_server_state(f"stage1_timeout_{port}", port)
        raise TimeoutError(f"Server on port {port} never opened TCP socket within {timeout}s. Last: {last_err}")

    # Stage 2: HTTP handlers respond to /v1/models.
    models_url = f"http://localhost:{port}/v1/models"
    stage2_start = time.time()
    probes = 0
    while time.time() < deadline:
        subprocess_died()
        try:
            rc = subprocess.run(
                ["curl","-sS","-o","/dev/null","-m","4","-w","%{http_code}", models_url],
                capture_output=True, text=True, timeout=6,
            )
            if rc.stdout.strip() == "200":
                _diag_log(f"wait_port_{port}",
                          f"stage2 /v1/models 200 at t={time.time()-t0:.1f}s (stage-local {time.time()-stage2_start:.1f}s, {probes+1} probes)")
                break
            last_err = f"/v1/models http={rc.stdout.strip()}"
        except subprocess.TimeoutExpired:
            last_err = "/v1/models curl hard-timeout"
        probes += 1
        if probes % 20 == 0:
            _diag_log(f"wait_port_{port}", f"stage2 still trying t={int(time.time()-t0)}s last={last_err}")
        time.sleep(3)
    else:
        _dump_server_state(f"stage2_timeout_{port}", port)
        raise TimeoutError(f"Server /v1/models never returned 200 within {timeout}s. Last: {last_err}")

    # Stage 3: warmup POST confirms the scheduler can generate.
    chat_url = f"http://localhost:{port}/v1/chat/completions"
    warmup_body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
    })
    stage3_start = time.time()
    probes = 0
    while time.time() < deadline:
        subprocess_died()
        try:
            rc = subprocess.run(
                ["curl","-sS","-o","/dev/null","-m","120","-w","%{http_code}",
                 "-H","Content-Type: application/json","-d", warmup_body, chat_url],
                capture_output=True, text=True, timeout=125,
            )
            if rc.stdout.strip() == "200":
                _diag_log(f"wait_port_{port}",
                          f"stage3 warmup POST 200 at t={time.time()-t0:.1f}s (stage-local {time.time()-stage3_start:.1f}s, {probes+1} probes). READY.")
                return
            last_err = f"warmup http={rc.stdout.strip()}"
        except subprocess.TimeoutExpired:
            last_err = "warmup curl hard-timeout"
        probes += 1
        if probes % 5 == 0:
            _diag_log(f"wait_port_{port}", f"stage3 still trying t={int(time.time()-t0)}s last={last_err}")
        time.sleep(5)
    _dump_server_state(f"stage3_timeout_{port}", port)
    raise TimeoutError(f"Server warmup POST never succeeded within {timeout}s. Last: {last_err}")


def _kill_pgroup(proc: subprocess.Popen) -> None:
    """Best-effort kill of the entire process group."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait()


@contextmanager
def server_context(launch_script: str, port: int, model_path: str):
    """Launch an SGLang server, yield when ready, and clean up on exit.

    Server stdout/stderr is tee'd to /logs/verifier/server_<port>.log so we
    can inspect live state even while the verifier is still running (pipes
    held by Popen are otherwise opaque until the process exits).

    Injects --skip-server-warmup into the sglang.launch_server CLI if the
    submission doesn't already set it. SGLang's internal warmup has a
    hardcoded 10-minute (600s) read timeout on its self-POST; heavy configs
    (fp8 + deep_gemm + speculative) exceed this and SGLang then kills its
    own server. Our Stage 3 warmup POST in wait_for_server already covers
    the same purpose with a longer budget.
    """
    env = {**os.environ, "PORT": str(port), "MODEL_PATH": model_path}
    try:
        os.makedirs(VERIFIER_LOG_DIR, exist_ok=True)
    except Exception:
        pass
    log_path = os.path.join(VERIFIER_LOG_DIR, f"server_{port}.log")

    # vLLM: no launch-script patching. vLLM handles its own startup warmup and CUDA-graph
    # capture; our three-stage wait_for_server (TCP -> /v1/models -> warmup POST) already
    # covers readiness on heavy configs (spec-decode + CUDA graphs). The launch script is
    # run verbatim so the candidate fully owns the server bring-up.
    patched_script = launch_script

    _diag_log(f"server_{port}", f"launching {patched_script}; stdout→{log_path}")
    log_fh = open(log_path, "w", buffering=1)  # line-buffered
    proc = subprocess.Popen(
        ["bash", patched_script],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _diag_log(f"server_{port}", f"bash pid={proc.pid}")
    try:
        wait_for_server(port, proc=proc)
        yield proc
    finally:
        _diag_log(f"server_{port}", "shutting down server")
        _kill_pgroup(proc)
        try:
            log_fh.close()
        except Exception:
            pass


# ===================================================================
# Benchmarking
# ===================================================================

def send_chat_request(port: int, messages: list, max_tokens: int) -> dict:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    resp = urlopen(req, timeout=REQUEST_TIMEOUT)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    body = json.loads(resp.read().decode())
    output_text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return {
        "total_ms": elapsed_ms,
        "output_text": output_text,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def _flush_cache(port: int) -> None:
    """Flush the server's KV cache between measurement rounds."""
    try:
        req = Request(
            f"http://localhost:{port}/flush_cache",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b"{}",
        )
        urlopen(req, timeout=10)
    except Exception:
        pass  # Not all servers support this endpoint.


def benchmark_server(
    port: int,
    workloads: list,
    *,
    warmup_override: int | None = None,
    measure_override: int | None = None,
) -> list:
    n_warmup = warmup_override if warmup_override is not None else WARMUP_ITERATIONS
    n_measure = measure_override if measure_override is not None else MEASURE_ITERATIONS
    results = []
    for wl in workloads:
        # Warmup (triggers CUDA graph capture, JIT, autotuning).
        for _ in range(n_warmup):
            send_chat_request(port, wl["messages"], wl["max_tokens"])
        # Flush KV cache so measurements start from clean state.
        _flush_cache(port)

        # Measure.
        measurements = []
        for _ in range(n_measure):
            result = send_chat_request(port, wl["messages"], wl["max_tokens"])
            measurements.append(result)

        latencies = [m["total_ms"] for m in measurements]

        results.append(
            {
                "name": wl["name"],
                "median_ms": statistics.median(latencies),
                "mean_ms": statistics.mean(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "stdev_ms": (
                    statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
                ),
                "all_ms": latencies,
            }
        )
    return results


def benchmark_server_concurrent(
    port: int,
    workloads: list,
    *,
    warmup_iterations: int = 10,
    measure_rounds: int = 10,
) -> list:
    """Benchmark with concurrent requests to test batching/scheduling.

    For each workload, sends `concurrency` simultaneous requests per round
    and measures per-request latency under load.
    """
    results = []
    for wl in workloads:
        concurrency = wl.get("concurrency", 1)
        # A workload may carry `mixed_messages` (a list of distinct message lists)
        # so each parallel slot gets its own request — this task's concurrent
        # workloads are HETEROGENEOUS bursts (short + long-prefill + medium mixed).
        # Falls back to `messages` broadcast to every slot.
        mixed = wl.get("mixed_messages")

        def _slot_messages(slot: int):
            if mixed:
                return mixed[slot % len(mixed)]
            return wl["messages"]

        # Warmup — send sequential requests to trigger CUDA graphs / JIT.
        for _ in range(warmup_iterations):
            send_chat_request(port, _slot_messages(0), wl["max_tokens"])
        _flush_cache(port)

        # Measure — send `concurrency` requests in parallel per round and record the
        # BURST MAKESPAN (wall-clock from dispatch to the last completion). Makespan is
        # the scheduler-relevant metric: it rewards admitting + batching the whole burst
        # efficiently (wide admission, chunked prefill, good interleave), not shrinking a
        # single request's latency by starving the batch. Per-request latency would
        # perversely reward a NARROW admission window (fewer concurrent seqs = each one
        # finishes sooner but the burst as a whole takes longer).
        makespans = []
        for _ in range(measure_rounds):
            slots = [_slot_messages(s) for s in range(concurrency)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(send_chat_request, port, msg, wl["max_tokens"])
                    for msg in slots
                ]
                ok = 0
                for fut in as_completed(futures):
                    try:
                        fut.result()
                        ok += 1
                    except Exception:
                        pass  # request failure under load
            makespan_ms = (time.perf_counter() - t0) * 1000.0
            # Only count a round where every request in the burst succeeded (a dropped
            # request would understate makespan).
            if ok == concurrency:
                makespans.append(makespan_ms)

        all_latencies = makespans

        if not all_latencies:
            results.append({
                "name": wl["name"],
                "median_ms": float("inf"),
                "mean_ms": float("inf"),
                "min_ms": float("inf"),
                "max_ms": float("inf"),
                "stdev_ms": 0.0,
                "all_ms": [],
                "concurrency": concurrency,
            })
            continue

        results.append(
            {
                "name": wl["name"],
                "median_ms": statistics.median(all_latencies),
                "mean_ms": statistics.mean(all_latencies),
                "min_ms": min(all_latencies),
                "max_ms": max(all_latencies),
                "stdev_ms": (
                    statistics.pstdev(all_latencies)
                    if len(all_latencies) > 1
                    else 0.0
                ),
                "all_ms": all_latencies,
                "concurrency": concurrency,
            }
        )
    return results


# ===================================================================
# Correctness
# ===================================================================

def load_prompts(path: Path) -> list[dict]:
    """Load JSONL prompts file."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(json.loads(line))
    return prompts


def collect_outputs(port: int, prompts: list[dict]) -> list[str | None]:
    """Run all prompts against a server and collect output texts.

    Aborts early if CONSECUTIVE_FAILURE_LIMIT consecutive requests fail
    (dead server protection).

    Diagnostic logging: per-prompt timing and cumulative pass/fail counts are
    appended to /logs/verifier/collect_port_<port>.log for live debugging.
    """
    outputs: list[str | None] = []
    failed = 0
    consecutive_failures = 0
    t_start = time.time()
    per_log = os.path.join(VERIFIER_LOG_DIR, f"collect_port_{port}.log")
    try:
        os.makedirs(VERIFIER_LOG_DIR, exist_ok=True)
        per_fh = open(per_log, "w", buffering=1)
        per_fh.write(f"# collect_outputs port={port} n_prompts={len(prompts)} started={time.strftime('%H:%M:%S')}\n")
        per_fh.write("# i\telapsed_ms\tstatus\tprompt_tokens\tcompletion_tokens\terror\n")
    except Exception:
        per_fh = None
    _diag_log(f"collect_port_{port}", f"begin n_prompts={len(prompts)}")
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        try:
            result = send_chat_request(port, prompt["messages"], prompt["max_tokens"])
            outputs.append(result["output_text"])
            consecutive_failures = 0
            if per_fh:
                per_fh.write(f"{i}\t{result['total_ms']:.0f}\tOK\t{result['prompt_tokens']}\t{result['completion_tokens']}\t-\n")
        except Exception as e:
            outputs.append(None)
            failed += 1
            consecutive_failures += 1
            dt_ms = (time.perf_counter() - t0) * 1000
            if per_fh:
                per_fh.write(f"{i}\t{dt_ms:.0f}\tFAIL\t-\t-\t{type(e).__name__}: {str(e)[:200]}\n")
            if failed <= 5:
                print(f"  WARN: prompt {i} failed: {e}")
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                msg = f"ABORT: {consecutive_failures} consecutive failures, stopping at {i + 1}/{len(prompts)}"
                print(f"  {msg}")
                _diag_log(f"collect_port_{port}", msg)
                _dump_server_state(f"collect_abort_{port}", port)
                break
        if (i + 1) % 250 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(prompts) - i - 1) / max(rate, 0.001)
            msg = (f"collected {i + 1}/{len(prompts)} "
                   f"(failed={failed}, rate={rate:.2f}/s, elapsed={elapsed:.0f}s, eta={eta:.0f}s)")
            print(f"  ... {msg}")
            _diag_log(f"collect_port_{port}", msg)
    if per_fh:
        try: per_fh.close()
        except Exception: pass
    elapsed = time.time() - t_start
    _diag_log(f"collect_port_{port}", f"done n_outputs={len(outputs)} failed={failed} elapsed={elapsed:.0f}s")
    print(f"  Collected {len(outputs)} outputs ({failed} failures)")
    return outputs


def compute_token_match(
    reference_outputs: list[str | None],
    candidate_outputs: list[str | None],
) -> dict:
    """Compare outputs token-by-token (whitespace-split words).

    Skips prompts where the reference failed.  Counts candidate failures as
    zero-match.

    Returns a dict with exact_match_rate, token_match_rate, and mismatch
    details for diagnostics.
    """
    exact_matches = 0
    token_ratios = []
    mismatches = []
    compared = 0

    for i, (ref, cand) in enumerate(zip(reference_outputs, candidate_outputs)):
        if ref is None:
            continue  # skip prompts that failed on baseline
        compared += 1

        if cand is None:
            token_ratios.append(0.0)
            mismatches.append({
                "index": i,
                "reason": "candidate request failed",
                "token_ratio": 0.0,
            })
            continue

        ref_norm = ref.strip()
        cand_norm = cand.strip()

        if ref_norm == cand_norm:
            exact_matches += 1
            token_ratios.append(1.0)
        else:
            ref_tokens = ref_norm.split()
            cand_tokens = cand_norm.split()

            # Count matching tokens from start (longest common prefix).
            prefix_matches = 0
            for rt, ct in zip(ref_tokens, cand_tokens):
                if rt == ct:
                    prefix_matches += 1
                else:
                    break

            max_len = max(len(ref_tokens), len(cand_tokens), 1)
            ratio = prefix_matches / max_len
            token_ratios.append(ratio)

            mismatches.append({
                "index": i,
                "ref_prefix": ref_norm[:100],
                "cand_prefix": cand_norm[:100],
                "ref_tokens": len(ref_tokens),
                "cand_tokens": len(cand_tokens),
                "prefix_match": prefix_matches,
                "token_ratio": round(ratio, 4),
            })

    avg_token_match = sum(token_ratios) / max(len(token_ratios), 1)

    return {
        "exact_match_rate": round(exact_matches / max(compared, 1), 4),
        "token_match_rate": round(avg_token_match, 4),
        "total_prompts": len(reference_outputs),
        "compared": compared,
        "exact_matches": exact_matches,
        "mismatches": mismatches[:30],  # cap for output readability
    }


# ===================================================================
# Scoring — reward.md 性能类, ABBA-paired MEDIAN speedup + bounded log envelope
# ===================================================================

def geometric_mean(values: list[float]) -> float:
    """Kept for DIAGNOSTIC reporting only. reward.md's `speedup` is the MEDIAN of the
    paired ratios, never the geomean — a sibling perf task archived a geomean as its
    ref_speedup and the two differed by 24% on the same 9 ratios."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def paired_ratios(
    baseline_1: list[dict], candidate: list[dict], baseline_2: list[dict] | None,
) -> tuple[list[float], list[dict]]:
    """One paired ratio per workload. The denominator is the MEAN of that workload's two
    baseline measurements when a third measurement phase exists (B-A-B), else the single one."""
    ratios: list[float] = []
    rows: list[dict] = []
    b2_by_name = {r["name"]: r for r in (baseline_2 or [])}
    for b1, c in zip(baseline_1, candidate):
        b2 = b2_by_name.get(b1["name"])
        b1_ms = b1["median_ms"]
        c_ms = c["median_ms"]
        if not math.isfinite(b1_ms) or not math.isfinite(c_ms) or c_ms <= 0 or b1_ms <= 0:
            rows.append({"name": b1["name"], "ratio": None, "baseline_1_ms": b1_ms,
                         "baseline_2_ms": (b2 or {}).get("median_ms"), "candidate_ms": c_ms,
                         "note": "non-finite timing; workload dropped from the median"})
            continue
        if b2 is not None and math.isfinite(b2["median_ms"]) and b2["median_ms"] > 0:
            denom = (b1_ms + b2["median_ms"]) / 2.0
            drift = abs(b1_ms - b2["median_ms"]) / denom
        else:
            denom = b1_ms
            drift = 0.0
        r = denom / c_ms
        ratios.append(r)
        rows.append({
            "name": b1["name"], "ratio": round(r, 6),
            "baseline_1_ms": round(b1_ms, 2),
            "baseline_2_ms": (round(b2["median_ms"], 2) if b2 else None),
            "baseline_paired_ms": round(denom, 2),
            "candidate_ms": round(c_ms, 2),
            "baseline_drift": round(drift, 4),
            "concurrency": b1.get("concurrency", 1),
        })
    return ratios, rows


def decode_work_probe(port: int) -> dict:
    """reward.md gate 3 'sleep 探针不减速' analogue for a SERVING task.

    A server that short-circuits generation (canned / cached / truncated output) does not
    pay for the tokens it claims. Ask the SAME prompt for 8 and then for 96 greedy tokens
    and require the wall clock to respond: lat(96) must exceed lat(8) by a clear margin
    AND the reported completion_tokens must grow. A non-responding server is cheating,
    not slow.
    """
    out: dict = {"probe": "decode-work responsiveness"}
    try:
        msgs = [{"role": "user", "content": "Count slowly from one to forty in words."}]
        small = [send_chat_request(port, msgs, 8) for _ in range(3)]
        large = [send_chat_request(port, msgs, 96) for _ in range(3)]
        s_ms = statistics.median(m["total_ms"] for m in small)
        l_ms = statistics.median(m["total_ms"] for m in large)
        s_tok = statistics.median(m["completion_tokens"] for m in small)
        l_tok = statistics.median(m["completion_tokens"] for m in large)
        out.update({"small_ms": round(s_ms, 2), "large_ms": round(l_ms, 2),
                    "small_completion_tokens": s_tok, "large_completion_tokens": l_tok,
                    "latency_ratio": round(l_ms / s_ms, 4) if s_ms > 0 else None,
                    "min_latency_ratio": DECODE_PROBE_MIN_RATIO})
        out["passed"] = bool(s_ms > 0 and (l_ms / s_ms) >= DECODE_PROBE_MIN_RATIO
                             and l_tok > s_tok)
    except Exception as exc:  # noqa: BLE001
        out.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return out


def pinned_ratio_check(ratios: list[float]) -> dict:
    """reward.md gate 3 '[1.0, 1.0, ...] 恒等 paired ratios'.

    Real hardware never returns identical paired ratios across heterogeneous workloads —
    the measured baseline-vs-itself spread on this task is 1-3% per workload. Ratios all
    pinned within PINNED_RATIO_TOL of 1.0 means the 'candidate' is the baseline server
    (or the timing path was bypassed), not a real optimisation.
    """
    if len(ratios) < 3:
        return {"passed": True, "note": "too few pairs to judge"}
    pinned = all(abs(r - 1.0) <= PINNED_RATIO_TOL for r in ratios)
    spread = max(ratios) - min(ratios)
    return {"passed": not pinned, "pinned": pinned, "tol": PINNED_RATIO_TOL,
            "spread": round(spread, 6),
            "note": "all paired ratios pinned at 1.0" if pinned else "ratios vary as real hardware does"}


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()

    cfg = load_reward_manifest()
    ref_speedup = manifest_ref_speedup(cfg)

    if args.fail:
        emit_reward(args.output_dir, 0.0, args.fail, args.total_time_ms,
                    hard_fail_reasons=[args.fail_gate], ref_speedup=ref_speedup)
        return

    if "__error__" in cfg:
        emit_reward(args.output_dir, 0.0,
                    f"frozen reward manifest unreadable at {REWARD_MANIFEST}: {cfg['__error__']}",
                    args.total_time_ms,
                    hard_fail_reasons=["build_or_entry_contract_failed"])
        return

    app_dir = Path(args.app_dir)
    model_path = str(app_dir / "model")
    candidate_launch = str(app_dir / "submission" / "launch_server.sh")
    baseline_launch = str(SCRIPT_DIR / "launch_baseline.sh")

    prompts = load_prompts(PROMPTS_PATH)
    print(f"Loaded {len(prompts)} correctness prompts")
    expected = cfg.get("expected_case_count")
    if isinstance(expected, int) and expected and len(prompts) != expected:
        emit_reward(args.output_dir, 0.0,
                    f"correctness prompt count {len(prompts)} != expected {expected}",
                    args.total_time_ms,
                    hard_fail_reasons=["forbidden_edit_path"], ref_speedup=ref_speedup)
        return

    match_result: dict = {}
    try:
        # --- Phase 1: Baseline (B) speed + reference outputs -----------------
        print("=" * 60)
        print("Phase 1: Launching baseline server (well-tuned config) ...")
        with server_context(baseline_launch, BASELINE_PORT, model_path):
            print(f"Baseline server ready on port {BASELINE_PORT}")
            baseline_results = benchmark_server(
                BASELINE_PORT, HIDDEN_WORKLOADS,
                warmup_override=SEQ_GATE_WARMUP, measure_override=SEQ_GATE_MEASURE)
            for r in baseline_results:
                print(f"  [baseline-1] {r['name']}: {r['median_ms']:.1f} ms")
            baseline_concurrent = benchmark_server_concurrent(
                BASELINE_PORT, CONCURRENT_WORKLOADS)
            for r in baseline_concurrent:
                print(f"  [baseline-1] {r['name']} (×{r['concurrency']}): {r['median_ms']:.1f} ms")

            print("\n--- Collecting reference outputs ---")
            reference_outputs = collect_outputs(BASELINE_PORT, prompts)
            ref_valid = sum(1 for o in reference_outputs if o is not None)
            print(f"  Reference: {ref_valid}/{len(prompts)} valid outputs")
            if ref_valid < MIN_VALID_OUTPUTS:
                emit_reward(args.output_dir, 0.0,
                            f"baseline only produced {ref_valid} valid outputs "
                            f"(need {MIN_VALID_OUTPUTS})",
                            args.total_time_ms,
                            hard_fail_reasons=["build_or_entry_contract_failed"],
                            ref_speedup=ref_speedup)
                return
        print("Baseline server stopped.\n")
        time.sleep(3)

        # --- Phase 2: Candidate (A) correctness + anti-cheat probes + speed ---
        print("=" * 60)
        print("Phase 2: Launching candidate server ...")
        with server_context(candidate_launch, CANDIDATE_PORT, model_path):
            print(f"Candidate server ready on port {CANDIDATE_PORT}")

            print("\n--- Collecting candidate outputs ---")
            candidate_outputs = collect_outputs(CANDIDATE_PORT, prompts)
            cand_valid = sum(1 for o in candidate_outputs if o is not None)
            print(f"  Candidate: {cand_valid}/{len(prompts)} valid outputs")

            print("\n--- Token-level correctness ---")
            match_result = compute_token_match(reference_outputs, candidate_outputs)
            print(f"  Compared: {match_result['compared']} prompts\n"
                  f"  Exact matches: {match_result['exact_matches']}"
                  f" ({match_result['exact_match_rate']:.1%})\n"
                  f"  Token match rate: {match_result['token_match_rate']:.4f}\n"
                  f"  Threshold: {TOKEN_MATCH_THRESHOLD}")
            if match_result["mismatches"]:
                n_shown = min(10, len(match_result["mismatches"]))
                print(f"\n  Sample mismatches ({n_shown} of {len(match_result['mismatches'])}):")
                for m in match_result["mismatches"][:n_shown]:
                    if "reason" in m:
                        print(f"    [{m['index']}] {m['reason']}")
                    else:
                        print(f"    [{m['index']}] ratio={m['token_ratio']:.3f} "
                              f"ref='{m['ref_prefix'][:60]}...' cand='{m['cand_prefix'][:60]}...'")

            if match_result["token_match_rate"] < TOKEN_MATCH_THRESHOLD:
                reason = (f"token match rate {match_result['token_match_rate']:.4f} "
                          f"below threshold {TOKEN_MATCH_THRESHOLD}")
                print(f"\nFAIL: {reason}")
                emit_reward(args.output_dir, 0.0, reason, args.total_time_ms,
                            hard_fail_reasons=["correctness_failed"],
                            additional_data={"correctness": match_result},
                            correctness=match_result, ref_speedup=ref_speedup)
                return
            print("\nPASS: token-level correctness\n")

            probe = decode_work_probe(CANDIDATE_PORT)
            print(f"--- decode-work probe: {json.dumps(probe)}")
            if not probe.get("passed"):
                emit_reward(args.output_dir, 0.0,
                            "candidate did not pay for the tokens it reported "
                            "(decode-work probe non-response)",
                            args.total_time_ms,
                            hard_fail_reasons=["cheating_detected"],
                            additional_data={"correctness": match_result, "decode_probe": probe},
                            correctness=match_result, ref_speedup=ref_speedup)
                return

            print("--- Benchmark ---")
            candidate_results = benchmark_server(
                CANDIDATE_PORT, HIDDEN_WORKLOADS,
                warmup_override=SEQ_GATE_WARMUP, measure_override=SEQ_GATE_MEASURE)
            for r in candidate_results:
                print(f"  [candidate] {r['name']}: {r['median_ms']:.1f} ms")
            candidate_concurrent = benchmark_server_concurrent(
                CANDIDATE_PORT, CONCURRENT_WORKLOADS)
            for r in candidate_concurrent:
                print(f"  [candidate] {r['name']} (×{r['concurrency']}): {r['median_ms']:.1f} ms")
        print("Candidate server stopped.\n")
        time.sleep(3)

        # --- Phase 3: Baseline re-check (the second B of B-A-B) --------------
        # BOTH the sequential anchors AND the concurrent bursts are re-measured, so every
        # one of the 6 workloads is a real ABBA pair (the previous shape re-checked only
        # the 3 sequential ones and paired the bursts against a single measurement).
        print("=" * 60)
        print("Phase 3: Baseline re-check ...")
        with server_context(baseline_launch, BASELINE_PORT, model_path):
            print(f"Baseline re-check server ready on port {BASELINE_PORT}")
            recheck_results = benchmark_server(
                BASELINE_PORT, HIDDEN_WORKLOADS,
                warmup_override=RECHECK_WARMUP, measure_override=RECHECK_ITERATIONS)
            for r in recheck_results:
                print(f"  [baseline-2] {r['name']}: {r['median_ms']:.1f} ms")
            recheck_concurrent = benchmark_server_concurrent(
                BASELINE_PORT, CONCURRENT_WORKLOADS,
                warmup_iterations=RECHECK_CONC_WARMUP, measure_rounds=RECHECK_CONC_ROUNDS)
            for r in recheck_concurrent:
                print(f"  [baseline-2] {r['name']} (×{r['concurrency']}): {r['median_ms']:.1f} ms")
        print("Baseline re-check stopped.\n")

        # --- Phase 4: paired median speedup -> bounded log reward ------------
        print("=" * 60)
        print("Phase 4: Computing score ...")
        # TIMED metric: the concurrent bursts only (the re-scoped responding set).
        ratios, rows = paired_ratios(
            baseline_concurrent, candidate_concurrent, recheck_concurrent)
        # GATE: the sequential single-request anchors, reported and gated but NOT scored.
        seq_ratios, seq_rows = paired_ratios(
            baseline_results, candidate_results, recheck_results)
        seq_regressions = [r for r in seq_rows
                           if r.get("ratio") is not None and r["ratio"] < SEQ_REGRESSION_MIN_RATIO]
        for row in seq_rows:
            print(f"  [latency-gate] {row['name']}: {row.get('ratio')}x "
                  f"(min {SEQ_REGRESSION_MIN_RATIO})")
        for row in rows:
            print(f"  {row['name']}: baseline {row.get('baseline_paired_ms')} ms "
                  f"(b1 {row.get('baseline_1_ms')} / b2 {row.get('baseline_2_ms')}) -> "
                  f"candidate {row.get('candidate_ms')} ms = {row.get('ratio')}x"
                  + (f"  [DRIFT {row['baseline_drift']:.0%}]"
                     if (row.get("baseline_drift") or 0) > VARIANCE_THRESHOLD else ""))

        variance_flags = [
            f"{row['name']}: baseline drift {row['baseline_drift']:.0%}"
            for row in rows if (row.get("baseline_drift") or 0) > VARIANCE_THRESHOLD]

        pinned = pinned_ratio_check(ratios)
        speedup = statistics.median(ratios) if ratios else None
        geo = geometric_mean(ratios) if ratios else None

        hard: list[str] = []
        if len(ratios) < MIN_PAIRS:
            hard.append("build_or_entry_contract_failed")
        if not pinned.get("passed", True):
            hard.append("cheating_detected")
        if seq_regressions:
            hard.append("sequential_latency_regression")

        if hard:
            reward = 0.0
            gate_reasons: list[str] = []
        else:
            reward, gate_reasons = bounded_log_reward(speedup, ref_speedup)
        hard.extend(gate_reasons)

        base_cvs = [row["baseline_drift"] for row in rows if row.get("baseline_drift") is not None]
        bench = {
            "speedup": speedup,
            "speedup_statistic": "median of the per-workload ABBA paired ratios",
            "ref_speedup": ref_speedup,
            "n_pairs": len(ratios),
            "paired_ratios": [row.get("ratio") for row in rows],
            "per_workload": rows,
            "diagnostic_geomean": geo,
            # 🔴 after the 2026-07-28 re-scope `ratios` holds ONLY the timed bursts, so these two
            # diagnostics must come from their own lists — the old index-slicing silently reported a
            # slice of the BURST list under the "sequential" label.
            "diagnostic_median_concurrent_only": (
                statistics.median(ratios) if ratios else None),
            "diagnostic_median_sequential_gate_only": (
                statistics.median(seq_ratios) if seq_ratios else None),
            "pinned_ratio_check": pinned,
            "variance_flags": variance_flags,
            "latency_regression_gate": {
                "min_ratio": SEQ_REGRESSION_MIN_RATIO,
                "per_workload": seq_rows,
                "regressions": [r["name"] for r in seq_regressions],
                "passed": not seq_regressions,
                "note": ("the sequential single-request anchors are a REGRESSION GATE, not part of "
                         "the timed median: they are structurally insensitive to scheduling "
                         "(0.96-1.05 measured across every configuration) so including them in the "
                         "median added noise with a vote, while excluding them entirely would let a "
                         "candidate win the bursts by tanking single-request latency."),
            },
        }
        if variance_flags:
            print("\nWARNING: High baseline variance detected:")
            for flag in variance_flags:
                print(f"  {flag}")
        print(f"\nPaired MEDIAN speedup: {speedup}  (diagnostic geomean {geo})")
        print(f"ref_speedup (frozen constant): {ref_speedup}")
        print(f"reward = {reward}   hard_fail_reasons={hard}")

        emit_reward(
            args.output_dir, reward,
            "benchmark complete" if not hard else "; ".join(hard),
            args.total_time_ms,
            subscores=rows,
            additional_data={"correctness": match_result, "variance_flags": variance_flags,
                             "decode_probe": probe, "pinned_ratio_check": pinned},
            hard_fail_reasons=hard, speedup=speedup, ref_speedup=ref_speedup,
            cv={"baseline": round(max(base_cvs), 4) if base_cvs else 0.0,
                "candidate": round(
                    statistics.pstdev(ratios) / speedup, 4)
                if (ratios and speedup and len(ratios) > 1) else 0.0},
            correctness=match_result, benchmark=bench)

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        emit_reward(args.output_dir, 0.0, f"verifier error: {exc}", args.total_time_ms,
                    hard_fail_reasons=["build_or_entry_contract_failed"],
                    additional_data={"traceback": traceback.format_exc()[-1500:]},
                    correctness=match_result, ref_speedup=ref_speedup)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — the 5-file contract must exist even on a hard crash
        traceback.print_exc()
        try:
            _out = "/logs/verifier"
            for _a in sys.argv:
                pass
            emit_reward(_out, 0.0, "verifier crashed before scoring", 0,
                        hard_fail_reasons=["build_or_entry_contract_failed"],
                        additional_data={"traceback": traceback.format_exc()[-1500:]})
        finally:
            raise SystemExit(1)
