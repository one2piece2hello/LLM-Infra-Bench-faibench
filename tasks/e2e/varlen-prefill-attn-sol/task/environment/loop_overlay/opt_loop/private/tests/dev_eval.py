#!/usr/bin/env python3
"""/opt/loop/private/tests/dev_eval.py — per-round DEV proxy for e2e-d1-varlen-prefill-attn-sol.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh. The solver's uid cannot read it.

WHAT IT DOES (a faithful, CHEAP, leak-free mirror of the graded path, on PUBLIC data):
  1. Imports the candidate's /app/repo/submission/varlen_prefill_attn.py VarlenPrefillAttention.
  2. Runs .forward over the PUBLIC dev suite (/opt/loop/private/tests/dev_suite.json, a byte copy of
     the model-visible /app/dev_bench/dev_suite.json: 3 cases + 3 degenerate edges), and enforces the
     REAL fp32 causal-parity correctness gate — the largest elementwise deviation from a float32
     causal-attention reference, divided by that row's RMS, must stay below a tolerance — the SAME
     parity measure the grade uses over its hidden rows.
  3. Times .forward with CUDA events and reports the geomean achieved TFLOP/s as the RAW best-of-k
     ranking signal. Higher TFLOP/s = a faster kernel; monotonic in the graded speedup.

WHAT IT NEVER TOUCHES / BAKES: the frozen harness (bench_prefill.py / baseline_prefill.py /
hidden_suite.json), the calibrated ref_speedup, or /tests. NO strong-baseline division, NO anchor:
dev_score is ABSOLUTE public-proxy TFLOP/s. The GRADED speedup divides by the HIDDEN strong baseline
over HIDDEN shapes; both rise with kernel throughput, so best-of-k ranking is faithful while the
absolute numbers differ in scale. Authoritative score = the runner end-of-session /tests grade.

REDUCED WORKLOAD (the GPU constraint): the dev suite is the author's own PUBLIC dev bench (3 cases +
3 edges, modest seq_lens), far smaller than the hidden scored suite; timing uses 6 reps x 4 inner
calls. A per-round pass completes in seconds on H20.

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

INV_SUBMISSION_MISSING = "submission_missing"
INV_LOAD = "varlen_prefill_attention_load_failed"
INV_PARITY = "causal_parity_failed"
INV_TIMING = "timing_invalid"
INV_HARNESS = "harness_error"

# fp32 causal-parity tolerance (relative-to-RMS). The graded harness uses the SAME measure; a
# correct bf16 causal attention lands FAR below this, a value of order 1 means the output is wrong.
PARITY_TOL = 0.25


def _cfg() -> dict:
    base = {"dev_suite": str(DEV_TESTS / "dev_suite.json"),
            "submission_path": "/app/repo/submission/varlen_prefill_attn.py",
            "timing_reps": 6, "inner_calls": 4, "parity_tol": PARITY_TOL}
    try:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        th = m.get("thresholds", m)
        for k in base:
            if k in th and th[k] is not None:
                base[k] = th[k]
    except Exception:
        pass
    return base


def _write_state(correctness_ok, failing_invariant, dev_score, dev_tflops, extra=None):
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {"correctness_ok": bool(correctness_ok), "hard_fail_reasons": reasons,
             "failing_invariant": failing_invariant or ""}
    if extra:
        state.update(extra)
    reward = {"dev_score": float(dev_score),
              "dev_metric": "achieved TFLOP/s (PUBLIC dev proxy; higher is better; NOT your graded score)",
              "dev_tflops": dev_tflops, "correctness_ok": bool(correctness_ok)}
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, 0.0, None, {"harness_message": msg})


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
        spec = importlib.util.spec_from_file_location("dev_varlen_impl", sub)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dev_varlen_impl"] = mod
        spec.loader.exec_module(mod)
        cls = getattr(mod, "VarlenPrefillAttention", None)
    except Exception as exc:
        _write_state(False, INV_LOAD, 0.0, None, {"detail": f"{type(exc).__name__}: {exc}"})
        return 0
    if cls is None:
        _write_state(False, INV_LOAD, 0.0, None, {"detail": "module has no VarlenPrefillAttention"})
        return 0

    def build(case):
        sl = [int(x) for x in case["seq_lens"]]
        Hq, Hkv, D = case["num_q_heads"], case["num_kv_heads"], case["head_size"]
        tot, mx = sum(sl), max(sl + [0])
        g = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
        mk = lambda h: (torch.randn(max(tot, 1), h, D, generator=g) * 0.5).to(
            torch.bfloat16).cuda()[:tot].contiguous()
        q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
        cu = torch.zeros(len(sl) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(torch.tensor(sl, dtype=torch.int32), 0)
        impl = cls({"num_q_heads": Hq, "num_kv_heads": Hkv, "head_size": D, "dtype": "bfloat16",
                    "device": "cuda", "max_num_seqs": len(sl), "max_seq_len": max(mx, 1),
                    "max_total_tokens": max(tot, 1), "causal": True,
                    "softmax_scale": 1.0 / math.sqrt(D)})
        impl.prepare()
        return impl, q, k, v, cu.cuda(), sl, Hq, Hkv, D, tot, mx

    def ref_row(q, k, v, aa, r, Hq, Hkv, D):
        rep = Hq // Hkv
        kk = k[aa:aa + r + 1].float(); vv = v[aa:aa + r + 1].float()
        if rep > 1:
            kk = kk.repeat_interleave(rep, dim=1); vv = vv.repeat_interleave(rep, dim=1)
        lo = torch.einsum("hd,shd->hs", q[aa + r].float(), kk) / math.sqrt(D)
        return torch.einsum("hs,shd->hd", torch.softmax(lo, -1), vv)

    tol = float(cfg["parity_tol"])
    tflops_list = []
    try:
        # edges: correctness-only (must not raise, must be parity-correct)
        for case in suite.get("edges", []):
            impl, q, k, v, cu, sl, Hq, Hkv, D, tot, mx = build(case)
            out = torch.empty(tot, Hq, D, device="cuda", dtype=torch.bfloat16)
            impl.forward(q, k, v, cu, mx, out)
            cul = [0]
            for s_ in sl:
                cul.append(cul[-1] + s_)
            worst = 0.0
            for i, s_ in enumerate(sl):
                for r in range(s_):
                    ref = ref_row(q, k, v, cul[i], r, Hq, Hkv, D)
                    rms = ref.pow(2).mean().sqrt().item()
                    worst = max(worst, (out[cul[i] + r].float() - ref).abs().max().item() / (rms + 1e-3))
            if worst > tol:
                _write_state(False, INV_PARITY, 0.0, None,
                             {"detail": f"edge {case.get('case_id')}: worst deviation {worst:.3f} > {tol}"})
                return 0
            del impl
            torch.cuda.empty_cache()

        for case in suite["cases"]:
            impl, q, k, v, cu, sl, Hq, Hkv, D, tot, mx = build(case)
            out = torch.empty(tot, Hq, D, device="cuda", dtype=torch.bfloat16)
            impl.forward(q, k, v, cu, mx, out)
            cul = [0]
            for s in sl:
                cul.append(cul[-1] + s)
            worst = 0.0
            for i, s in enumerate(sl):
                for r in (0, s // 2, s - 1):
                    ref = ref_row(q, k, v, cul[i], r, Hq, Hkv, D)
                    rms = ref.pow(2).mean().sqrt().item()
                    worst = max(worst, (out[cul[i] + r].float() - ref).abs().max().item() / (rms + 1e-3))
            if worst > tol:
                _write_state(False, INV_PARITY, 0.0, None,
                             {"detail": f"case {case.get('case_id')}: worst deviation {worst:.3f} > {tol}"})
                return 0
            reps = int(cfg["timing_reps"]); inner = int(cfg["inner_calls"])
            for _ in range(3):
                impl.forward(q, k, v, cu, mx, out)
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                e0.record()
                for _ in range(inner):
                    impl.forward(q, k, v, cu, mx, out)
                e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / 1e3 / inner)
            ts.sort()
            t = ts[len(ts) // 2]
            if t <= 0:
                _write_state(False, INV_TIMING, 0.0, None, {"detail": "non-positive timing"})
                return 0
            flops = sum(4.0 * Hq * D * (s * (s + 1) / 2.0) for s in sl)
            tflops_list.append(flops / t / 1e12)
            del impl, q, k, v, out
            torch.cuda.empty_cache()
    except Exception as exc:
        import traceback
        _write_state(False, INV_TIMING, 0.0, None,
                     {"detail": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-600:]})
        return 0

    if not tflops_list:
        _write_state(False, INV_TIMING, 0.0, None, {"detail": "no cases timed"})
        return 0
    logsum = sum(math.log(x) for x in tflops_list if x > 0)
    dev_tflops = math.exp(logsum / len([x for x in tflops_list if x > 0]))
    _write_state(True, None, dev_tflops, dev_tflops,
                 {"detail": f"fp32 causal parity OK on {len(suite['cases'])} cases + "
                            f"{len(suite.get('edges', []))} edges; geomean {dev_tflops:.2f} TFLOP/s"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _harness_error(f"dev_eval crashed: {type(exc).__name__}: {exc}")
        raise SystemExit(0)
