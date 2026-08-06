"""Verifier core — correctness gate (golden recomputed from oracle each run) + isolated timed
benchmark + reward. Reviewer/verifier-only.

Reward (IMPLEMENTATION class -> BINARY):
    reward = 1.0  iff EVERY hidden case passes and no cheat/hard-fail;
    reward = 0.0  otherwise (a single failing check scores 0).
task_class is 实现类. This is implement-from-empty: the starter is an
empty stub raising NotImplementedError so noop fails correctness, there is NO timeable
baseline, and the only timing anchor is the oracle itself (vs_oracle ~= 1.0 ->
ln(ref) ~= 0), which makes the performance log formula degenerate. vs_oracle is still
measured and reported as DIAGNOSTIC METADATA, never as the reward.

MODE (env WRE_MODE): candidate | noop | oracle | baseline2 | negative
The scored module path is passed in; goldens are recomputed from the ORACLE every run (§A3) —
nothing to grep/copy. All three functions are checked on every hidden case.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

# Pin single-thread BEFORE importing numpy (and torch below) so wall-clock oracle_ms is
# co-tenancy-stable: the baked oracle_ms otherwise swings widely (observed ~36x on shared CPU)
# when numpy's BLAS / torch's intra-op pool fan out across cores contended by neighbors. The
# numpy BLAS/OpenMP backends read these env vars at import, so they MUST precede `import numpy`;
# torch is pinned via set_num_threads(1) right after it is imported. This task uses BOTH libs.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workload as W  # noqa: E402

ATOL = 2e-4
RTOL = 1e-3
LAT_STABILITY_MAX = 8.0  # per-iter max/min guard (anti "does less work later")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cases_equal(a, b):
    a = a.detach().to(torch.float32); b = b.detach().to(torch.float32)
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if torch.isnan(a).any() or torch.isinf(a).any():
        return False, "nan/inf in candidate"
    ok = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    if not ok:
        d = (a - b).abs().max().item()
        return False, f"max_abs_diff={d:.3e}"
    return True, ""


def check_correctness(cand, oracle):
    """Run all 3 functions on every hidden case; return (passed, details)."""
    checks = []
    for spec in W.HIDDEN_CASES:
        wl = W.make_workload(spec)
        r, m, idx = wl["token_level_rewards"], wl["response_mask"], wl["index"]
        # as_torch_index + group_mean_std probed directly
        try:
            g_c = cand.as_torch_index(idx); g_o = oracle.as_torch_index(idx)
            ok, why = _cases_equal(g_c.float(), g_o.float())
            checks.append({"case": spec["name"], "fn": "as_torch_index", "passed": ok, "why": why})
            sc = r.sum(-1)
            mc = cand.group_mean_std(sc, g_c); mo = oracle.group_mean_std(sc, g_o)
            for k, (cc, oo) in enumerate(zip(mc, mo)):
                ok, why = _cases_equal(cc, oo)
                checks.append({"case": spec["name"], "fn": f"group_mean_std[{k}]", "passed": ok, "why": why})
            for fn in ("compute_grpo_outcome_advantage", "compute_rloo_outcome_advantage"):
                ca = getattr(cand, fn)(r, m, idx)[0]
                oa = getattr(oracle, fn)(r, m, idx)[0]
                ok, why = _cases_equal(ca, oa)
                checks.append({"case": spec["name"], "fn": fn, "passed": ok, "why": why})
        except NotImplementedError as e:
            checks.append({"case": spec["name"], "fn": "?", "passed": False, "why": f"NotImplementedError:{e}"})
        except Exception as e:
            checks.append({"case": spec["name"], "fn": "?", "passed": False, "why": f"{type(e).__name__}:{e}"})
    # metamorphic: permutation-equivariance of GRPO on one case
    try:
        wl = W.make_workload(W.HIDDEN_CASES[0]); r, m, idx = wl["token_level_rewards"], wl["response_mask"], wl["index"]
        perm = torch.randperm(r.shape[0], generator=torch.Generator().manual_seed(7))
        a1 = cand.compute_grpo_outcome_advantage(r, m, idx)[0][perm]
        a2 = cand.compute_grpo_outcome_advantage(r[perm], m[perm], idx[perm.numpy()])[0]
        ok, why = _cases_equal(a1, a2)
        checks.append({"case": "metamorphic_perm", "fn": "grpo", "passed": ok, "why": why})
    except Exception as e:
        checks.append({"case": "metamorphic_perm", "fn": "grpo", "passed": False, "why": f"{type(e).__name__}:{e}"})
    passed = all(c["passed"] for c in checks)
    return passed, checks


def _bench_once(fn, args):
    t0 = time.perf_counter(); fn(*args); return (time.perf_counter() - t0) * 1e3


def bench(mod, warmup=5, blocks=3, iters=40):
    """Isolated timed region: only the estimator calls on the perf case. Block-of-medians."""
    spec = next(c for c in W.HIDDEN_CASES if c["name"] == "perf_bigbatch")
    wl = W.make_workload(spec); r, m, idx = wl["token_level_rewards"], wl["response_mask"], wl["index"]
    def run():
        mod.compute_grpo_outcome_advantage(r, m, idx)
        mod.compute_rloo_outcome_advantage(r, m, idx)
    for _ in range(warmup):
        run()
    block_meds, alls = [], []
    for _ in range(blocks):
        ts = sorted(_bench_once(run, ()) for _ in range(iters))
        alls += ts
        trim = ts[int(0.1 * iters): max(int(0.1 * iters) + 1, int(0.9 * iters))]
        block_meds.append(sum(trim) / len(trim))
    ms = sorted(block_meds)[len(block_meds) // 2]
    stability = (max(alls) / max(min(alls), 1e-9))
    return ms, stability


def main():
    mode = os.environ.get("WRE_MODE", "candidate")
    scored_path = os.environ["WRE_SCORED_MODULE"]      # candidate submission path (mode-dispatched by test.sh)
    oracle_path = os.environ["WRE_ORACLE_MODULE"]      # reviewer-only oracle
    out = os.environ.get("WRE_OUT", "/tmp/wre_result.json")

    oracle = _load(oracle_path, "wre_oracle")
    cand = _load(scored_path, "wre_cand")

    result = {"mode": mode, "reward": 0.0, "correctness_passed": False}
    passed, checks = check_correctness(cand, oracle)
    result["correctness_passed"] = passed
    result["n_checks"] = len(checks)
    result["failed_checks"] = [c for c in checks if not c["passed"]][:8]
    n_failed = sum(1 for c in checks if not c["passed"])
    result["task_type"] = "implementation"
    result["reward_formula"] = "implementation: 1.0 iff every case passes and no cheat, else 0.0"
    result["tests"] = {"passed": len(checks) - n_failed, "total": len(checks)}
    if not passed:
        result["reason"] = "correctness_fail"
        result["hard_fail_reasons"] = ["correctness_failed"]
        _emit(result, out); return

    # BINARY reward: every case passed and no cheat => 1.0. Timing is still measured
    # and reported as diagnostic metadata (vs_oracle), but is NOT the reward.
    result["reward"] = 1.0
    result["hard_fail_reasons"] = []
    try:
        cand_ms, stab = bench(cand)
        orac_ms, _ = bench(oracle)
        result["candidate_ms"] = cand_ms
        result["oracle_ms"] = orac_ms
        result["latency_stability"] = stab
        vs_oracle = orac_ms / max(cand_ms, 1e-9)
        result["speedup"] = float(vs_oracle)
        result["vs_oracle"] = float(vs_oracle)
        result["metadata"] = {"vs_oracle": float(vs_oracle), "candidate_ms": cand_ms,
                              "oracle_ms": orac_ms, "latency_stability": stab,
                              "note": "vs_oracle is diagnostic metadata only, never the reward"}
        # Anti-cheat kept HARD: a wildly unstable per-iter profile is the classic
        # "does less work later" signature and still zeroes the reward.
        if stab > LAT_STABILITY_MAX:
            result["reason"] = f"unstable_latency {stab:.1f}>{LAT_STABILITY_MAX}"
            result["reward"] = 0.0
            result["hard_fail_reasons"] = ["unstable_latency"]
    except Exception as exc:
        # Timing-only infra failure: RECORD it, but do not zero a fully-correct solve
        # (timing is not the metric for this implementation-class task).
        result["timing_error"] = f"{type(exc).__name__}: {exc}"
    _emit(result, out)


def _emit(result, out):
    with open(out, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
