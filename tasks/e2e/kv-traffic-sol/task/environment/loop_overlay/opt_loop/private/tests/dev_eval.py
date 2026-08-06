#!/usr/bin/env python3
"""/opt/loop/private/tests/dev_eval.py — per-round DEV proxy for e2e-b1-kv-traffic-sol.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh. The solver's uid cannot read it.

WHAT IT DOES (a faithful, CHEAP, leak-free mirror of the graded path, on PUBLIC data):
  1. Imports the candidate's /app/repo/submission/kv_traffic.py KVTrafficEngine.
  2. Runs scatter -> poison -> gather on the PUBLIC dev suite (/opt/loop/private/tests/dev_suite.json,
     a byte-identical copy of the model-visible /app/dev_bench/dev_suite.json: 3 tiny cases), and
     enforces the REAL bit-exact KV round-trip correctness gate (torch.equal of every gathered layer
     against the scattered source) — the SAME correctness invariant the grade's hidden suite checks.
  3. Times gather+scatter with CUDA events and reports the geomean achieved bandwidth (GB/s) as the
     RAW best-of-k ranking signal. Higher GB/s = a faster engine; monotonic in the graded speedup.

WHAT IT NEVER TOUCHES / BAKES: the frozen harness (bench_kvtraffic.py / baseline_kv_traffic.py /
hidden_suite.json), the calibrated ref_speedup, or /tests. There is NO strong-baseline division and
NO anchor: dev_score is an ABSOLUTE public-proxy bandwidth, never normalized against the grade. The
GRADED speedup divides by the HIDDEN strong baseline over HIDDEN shapes; this dev signal is raw
bandwidth on the public dev shapes. Both rise with engine throughput, so best-of-k ranking is
faithful; the absolute numbers differ in scale. Authoritative score = the runner end-of-session
/tests grade.

REDUCED WORKLOAD (the GPU constraint): the dev suite is the author's own PUBLIC dev bench (3 small
cases, few layers), far smaller than the hidden scored suite, and timing uses 8 reps — a per-round
pass completes in seconds on H20.

OUTPUT: /logs/loop/dev/{verifier_state.json, reward.json}; on infra failure, harness_error.txt.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

LOOP_PRIVATE = Path("/opt/loop/private")
MANIFEST = LOOP_PRIVATE / "manifest.json"
DEV_TESTS = LOOP_PRIVATE / "tests"
DEV_OUT = Path("/logs/loop/dev")
DEV_OUT.mkdir(parents=True, exist_ok=True)

POISON = -12345.0

INV_SUBMISSION_MISSING = "submission_missing"
INV_LOAD = "kv_traffic_engine_load_failed"
INV_ROUNDTRIP = "kv_roundtrip_not_bit_exact"
INV_TIMING = "timing_invalid"
INV_HARNESS = "harness_error"


def _cfg() -> dict:
    base = {
        "dev_suite": str(DEV_TESTS / "dev_suite.json"),
        "submission_path": "/app/repo/submission/kv_traffic.py",
        "timing_reps": 8,
    }
    try:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        th = m.get("thresholds", m)
        for k in base:
            if k in th and th[k] is not None:
                base[k] = th[k]
    except Exception:
        pass
    return base


def _write_state(correctness_ok, failing_invariant, dev_score, dev_gbps, extra=None):
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {"correctness_ok": bool(correctness_ok), "hard_fail_reasons": reasons,
             "failing_invariant": failing_invariant or ""}
    if extra:
        state.update(extra)
    reward = {"dev_score": float(dev_score),
              "dev_metric": "achieved bandwidth GB/s (PUBLIC dev proxy; higher is better; NOT your graded score)",
              "dev_gbps": dev_gbps, "correctness_ok": bool(correctness_ok)}
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, 0.0, None, {"harness_message": msg})


def _load_engine(path: Path):
    spec = importlib.util.spec_from_file_location("dev_kv_impl", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dev_kv_impl"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    cfg = _cfg()
    sub = Path(cfg["submission_path"])
    if not sub.exists():
        _write_state(False, INV_SUBMISSION_MISSING, 0.0, None, {"detail": f"{sub} not found"})
        return 0
    try:
        import torch
    except Exception as exc:
        _harness_error(f"torch import failed: {type(exc).__name__}: {exc}")
        return 0
    if not torch.cuda.is_available():
        _harness_error("CUDA not available for the dev proxy")
        return 0
    try:
        suite = json.loads(Path(cfg["dev_suite"]).read_text())
    except Exception as exc:
        _harness_error(f"could not read dev suite: {type(exc).__name__}: {exc}")
        return 0
    try:
        mod = _load_engine(sub)
    except Exception as exc:
        # a bad import here is a candidate defect, not infra.
        _write_state(False, INV_LOAD, 0.0, None, {"detail": f"{type(exc).__name__}: {exc}"})
        return 0
    if not hasattr(mod, "KVTrafficEngine"):
        _write_state(False, INV_LOAD, 0.0, None, {"detail": "module has no KVTrafficEngine"})
        return 0

    dev = "cuda"
    gbps_list = []
    try:
        for cfg_case in suite["cases"]:
            B, L, P = cfg_case["batch"], cfg_case["num_layers"], cfg_case["page_size"]
            Hkv, D, NP = cfg_case["num_kv_heads"], cfg_case["head_size"], cfg_case["num_pages"]
            dt = torch.bfloat16 if cfg_case.get("dtype", "bfloat16") == "bfloat16" else torch.float16
            g = torch.Generator(device="cpu").manual_seed(cfg_case["seed"])
            seqs = list(cfg_case["seq_lens"])
            need = [(s + P - 1) // P for s in seqs]
            mp = max(need)
            order = torch.randperm(NP, generator=g)
            bt = torch.full((B, mp), -1, dtype=torch.int32)
            cur = 0
            for b in range(B):
                bt[b, :need[b]] = order[cur:cur + need[b]].to(torch.int32)
                cur += need[b]
            K = [[(torch.randn(seqs[b], Hkv, D, generator=g) * 0.5).to(dt).to(dev) for b in range(B)]
                 for _ in range(L)]
            V = [[(torch.randn(seqs[b], Hkv, D, generator=g) * 0.5).to(dt).to(dev) for b in range(B)]
                 for _ in range(L)]
            build = {"num_layers": L, "num_kv_heads": Hkv, "head_size": D, "page_size": P,
                     "num_pages": NP, "max_batch": B, "max_pages_per_request": mp,
                     "dtype": cfg_case.get("dtype", "bfloat16"), "device": dev}
            eng = mod.KVTrafficEngine(build)
            eng.allocate()
            torch.cuda.synchronize()
            ctx = torch.zeros(B, dtype=torch.int32)
            new = torch.tensor(seqs, dtype=torch.int32)
            T = int(new.sum())
            plan = {"block_table": bt.to(dev), "ctx_lens": ctx.to(dev), "new_lens": new.to(dev),
                    "block_table_cpu": bt, "ctx_lens_cpu": ctx, "new_lens_cpu": new,
                    "total_tokens": T, "batch": B}
            src = [(torch.cat([K[l][b] for b in range(B)]).contiguous(),
                    torch.cat([V[l][b] for b in range(B)]).contiguous()) for l in range(L)]
            kout = [torch.empty(T, Hkv, D, dtype=dt, device=dev) for _ in range(L)]
            vout = [torch.empty_like(kout[0]) for _ in range(L)]

            def do_scatter():
                eng.begin_step(plan)
                for l in range(L):
                    eng.scatter(l, src[l][0], src[l][1])

            def do_gather():
                eng.begin_step(plan)
                for l in range(L):
                    eng.gather(l, kout[l], vout[l])

            # correctness: bit-exact round trip (the REAL graded invariant)
            do_scatter()
            for l in range(L):
                kout[l].fill_(POISON)
                vout[l].fill_(POISON)
            do_gather()
            torch.cuda.synchronize()
            ok = all(torch.equal(kout[l], src[l][0]) and torch.equal(vout[l], src[l][1])
                     for l in range(L))
            if not ok:
                _write_state(False, INV_ROUNDTRIP, 0.0, None,
                             {"detail": f"case {cfg_case['case_id']}: gathered KV != scattered source"})
                return 0

            def timed(fn, reps):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1) / 1e3)
                ts.sort()
                return ts[len(ts) // 2]

            reps = int(cfg["timing_reps"])
            elt = 2
            by = 2 * 2 * L * T * Hkv * D * elt
            tg = timed(do_gather, reps)
            ts_ = timed(do_scatter, reps)
            if tg <= 0 or ts_ <= 0:
                _write_state(False, INV_TIMING, 0.0, None, {"detail": "non-positive timing"})
                return 0
            gbps_list.append(by / tg / 1e9)
            gbps_list.append(by / ts_ / 1e9)
            del eng, K, V, src, kout, vout
            torch.cuda.empty_cache()
    except Exception as exc:
        import traceback
        _write_state(False, INV_TIMING, 0.0, None,
                     {"detail": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-600:]})
        return 0

    if not gbps_list:
        _write_state(False, INV_TIMING, 0.0, None, {"detail": "no cases timed"})
        return 0
    # geomean achieved bandwidth across the (gather, scatter) x cases measurements
    logsum = sum(math.log(x) for x in gbps_list if x > 0)
    dev_gbps = math.exp(logsum / len(gbps_list))
    _write_state(True, None, dev_gbps, dev_gbps,
                 {"detail": f"bit-exact on {len(suite['cases'])} public dev cases; "
                            f"geomean {dev_gbps:.1f} GB/s over {len(gbps_list)} measurements"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _harness_error(f"dev_eval crashed: {type(exc).__name__}: {exc}")
        raise SystemExit(0)
