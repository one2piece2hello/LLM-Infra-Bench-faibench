"""e2e-a3-moe-train-budget verifier (FROZEN eval surface).

REWARD MODEL (the bench reward spec — performance class, BOUNDED log form):

    speedup     = baseline_bpb / candidate_val_bpb        # quality ratio at a FIXED budget
    ref_speedup = baseline_bpb / oracle_val_bpb           # FROZEN authoring constant
    reward      = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0          in [0, 1]

`baseline_bpb` and `ref_speedup` are **frozen constants** carried in the frozen manifest
(`reward_model` block) and calibrated at authoring time on NVIDIA H20. This scorer NEVER runs
the oracle and NEVER runs the baseline: it reads the two constants and measures only the
candidate. There is no oracle/reference recipe anywhere in the task image.

Semantics: `speedup == ref_speedup => 0.0 (must EXCEED it to score) ; `speedup >= ref_speedup**2` => capped 1.0 ;
`speedup <= 1` (did not beat the baseline recipe it started from) => 0 via pre-gate 5.
Matching the *baseline* therefore scores **0**, not 1.0 — the old open-ended
"matching the baseline scores 1.0, no upper cap" reading is retired.

The metric is a *quality* metric at a fixed training budget, not a latency measurement:
no wall-clock timing enters the reward (`timing_measured: false`). `"speedup"` is kept as
the field name for schema compatibility with the rest of the bench and is described by the
companion fields `metric_kind` / `metric_name` / `metric_direction`.

EVAL-ONLY. `test.sh` has already run the solver's training under the harness-owned
wall-clock timer (external process-group kill — never a cooperative in-loop timer, never a
self-reported step/token count) and produced a checkpoint. This script:
  1. loads that checkpoint via the solver's `load_model_for_verification(path, device)`
     hook in an isolated process (never re-trains, never trusts a self-reported number);
  2. RE-COUNTS the model's TOTAL parameters and enforces the hard capacity FLOOR
     (P_total >= param_floor) — the MoE-forcing constraint, re-measured by the harness
     from the loaded module;
  3. runs the parameter-golf anti-spoof kit (forward(x)/forward(x,y) CE agreement,
     position-variance floor, OOD real-vs-arbitrary);
  4. computes a stabilized median-of-3-shifts val_bpb on the HIDDEN WikiText-103 TEST
     split (`/tests/holdout_corpus.txt`, uploaded fresh at scoring — NOT baked);
  5. applies the six reward.md performance pre-gates with NAMED reasons and emits the
     bounded reward plus the 5-file `/logs/verifier` contract.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))
CKPT_PATH = Path(os.environ.get("OUT_CKPT", str(SUBMISSION_DIR / "model_ckpt.pt")))

# The frozen surface lives at /tests ONLY. There is deliberately NO /opt/verifier fallback:
# a baked fallback copy is (a) a stale-scoring hazard and (b) an in-image leak of the
# held-out corpus and the calibrated constants.
FROZEN_DIR = Path("/tests")
MANIFEST_PATH = FROZEN_DIR / "verifier-correctness-manifest.json"
HOLDOUT_CORPUS_PATH = FROZEN_DIR / "holdout_corpus.txt"
DEFAULT_TOKENIZER_PATH = Path("/data/tokenizers/wikitext_bpe8k.model")
DEFAULT_SEQ_LEN = 512
DEFAULT_VAL_BATCH_TOKENS = 32_768

VERIFIER_DIR = Path("/logs/verifier")
# The verifier output contract ("5-file contract" — 6 files in the corrected bench-wide form,
# identical to the embed-compress-golf sibling). Emitted on EVERY exit path, hard fails and
# crashes included.
RESULT_FILES = (
    "reward.txt",
    "reward.json",
    "metrics.json",
    "verifier_state.json",
    "correctness_results.json",
    "benchmark_results.json",
)

# ---- named hard-fail reasons (reward.md performance pre-gates 1..6) ----
# Vocabulary shared with the embed-compress-golf / quant-golf siblings so downstream audits
# can group reasons across the bench.
R_BUILD = "build_or_entry_contract_failed"          # gate 1
R_CORRECT = "correctness_failed"                    # gate 2
R_CHEAT = "anti_cheat_triggered"                    # gate 3
R_FORBIDDEN = "forbidden_edit_path"                 # gate 4
R_NO_GAIN = "speedup_not_above_baseline"            # gate 5  (speedup <= 1)
R_BAD_REF = "ref_speedup_invalid_or_missing"        # gate 6  (ref_speedup <= 1 / missing)
R_CRASH = "verifier_crashed"
R_UNNAMED = "zero_without_named_reason"             # backstop: reward 0 must always be explained
R_INCOMPLETE = "gate_suite_incomplete"              # a truncated suite can never pass

REF_SPEEDUP_AUTHORING_FLOOR = 1.15   # reward.md authoring legality floor

# A truncated check suite (early return after a missing checkpoint / an unloadable model /
# a probe exception) yields FEWER checks than this and can never score above 0.
#   1 import+reload, 2 real-LM param floor, 3 A3 capacity floor, 4 CE agreement,
#   5 plausibility band, 6 OOD real-vs-arbitrary
EXPECTED_TOTAL_CHECKS = 6


# -----------------------------------------------------------------------------
# frozen manifest (constants only — never an oracle run)
# -----------------------------------------------------------------------------
def load_manifest() -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    if not MANIFEST_PATH.exists():
        problems.append(f"frozen manifest missing at {MANIFEST_PATH}")
        return {}, problems
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), problems
    except Exception as exc:
        problems.append(f"frozen manifest unreadable: {type(exc).__name__}: {exc}")
        return {}, problems


_MAN, _MAN_PROBLEMS = load_manifest()
_RM: dict[str, Any] = _MAN.get("reward_model") or {}
_QG: dict[str, Any] = _MAN.get("quality_gate") or {}

# FROZEN reward constants. `None` => fail closed with a named reason; never defaulted to a
# guess, never recomputed by running a reference recipe.
BASELINE_BPB = _RM.get("baseline_bpb")
REF_SPEEDUP = _RM.get("ref_speedup")
BASELINE_CV = _RM.get("baseline_cv")

PARAM_FLOOR = int(_QG.get("param_floor", _MAN.get("param_floor", 150_000_000)))
MIN_PARAM_COUNT = int(_QG.get("min_param_count", 1_000_000))
MIN_PLAUSIBLE_VAL_BPB = float(_QG.get("min_plausible_val_bpb", 0.90))
MAX_PLAUSIBLE_VAL_BPB = float(_QG.get("max_plausible_val_bpb", 3.0))
OOD_LOSS_RATIO = float(_QG.get("ood_loss_ratio", 0.7))
LOGITS_POSITION_VARIANCE_FLOOR = float(_QG.get("logits_position_variance_floor", 1e-4))
LOGITS_LOSS_AGREEMENT_TOL = float(_QG.get("logits_loss_agreement_tol", 5e-2))


class Check:
    def __init__(self, name, passed, message, details=None, gate=R_CORRECT):
        self.name, self.passed, self.message = name, passed, message
        self.details = details or {}
        self.gate = gate


def result(name, passed, message, gate=R_CORRECT, **details):
    return Check(name, passed, message, details, gate)


# -----------------------------------------------------------------------------
# 5-file emitter (single writer, so hard-fail and success paths cannot diverge)
# -----------------------------------------------------------------------------
def emit(reward: float,
         hard_fail_reasons: list[str],
         checks: list[Check] | None = None,
         speedup: float | None = None,
         val_bpb: float | None = None,
         bpb_passes: list[float] | None = None,
         extra: dict[str, Any] | None = None) -> None:
    checks = checks or []
    reward = float(reward)
    if not math.isfinite(reward):
        reward = 0.0
        hard_fail_reasons = list(hard_fail_reasons) + [R_UNNAMED]
    reward = min(1.0, max(0.0, reward))                    # reward MUST stay in [0, 1]
    reasons = [r for i, r in enumerate(hard_fail_reasons) if r not in hard_fail_reasons[:i]]
    if reward == 0.0 and not reasons:
        reasons = [R_UNNAMED]                              # never a silent zero
    if reasons:
        reward = 0.0

    npass = sum(1 for c in checks if c.passed)
    total = len(checks)
    cand_cv = None
    if bpb_passes and len(bpb_passes) >= 2:
        m = statistics.fmean(bpb_passes)
        if m > 0:
            cand_cv = statistics.pstdev(bpb_passes) / m

    reward_json = {
        "task_type": "performance",
        "reward": reward,
        "hard_fail_reasons": reasons,
        "speedup": speedup,
        "ref_speedup": REF_SPEEDUP,
        "cv": {"baseline": BASELINE_CV, "candidate": cand_cv},
        # descriptive companions: this is a QUALITY ratio at a fixed training budget, and
        # no wall-clock timing enters the reward. "speedup" is retained for schema
        # compatibility with the rest of the bench.
        "metric_kind": "quality_at_fixed_budget",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "timing_measured": False,
    }
    correctness = {
        "passed": npass,
        "total": total,
        "expected_total": EXPECTED_TOTAL_CHECKS,
        "all_passed": total == EXPECTED_TOTAL_CHECKS and npass == total,
        "hard_fail_reasons": reasons,
        "checks": [{"name": c.name, "passed": c.passed, "message": c.message,
                    "gate": c.gate, "details": c.details} for c in checks],
    }
    benchmark = {
        "metric_kind": "quality_at_fixed_budget",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "timing_measured": False,
        "candidate_val_bpb": val_bpb,
        "candidate_val_bpb_shift_passes": bpb_passes,
        "baseline_bpb_frozen": BASELINE_BPB,
        "oracle_bpb_frozen": (
            None if (not isinstance(BASELINE_BPB, (int, float)) or not isinstance(REF_SPEEDUP, (int, float))
                     or not REF_SPEEDUP) else float(BASELINE_BPB) / float(REF_SPEEDUP)),
        "ref_speedup_frozen": REF_SPEEDUP,
        "speedup": speedup,
        "oracle_executed_by_scorer": False,
        "baseline_executed_by_scorer": False,
    }
    metrics = {
        "reward": reward,
        "task_type": "performance",
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "reward_spec": "the bench reward spec (performance class, bounded log form)",
        "reward_form": (
            "reward.md 性能类: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 in [0,1]; "
            "speedup = baseline_bpb / candidate_val_bpb at a FIXED wall-clock training budget; "
            "ref_speedup is a FROZEN authoring-time constant (baseline_bpb / oracle_val_bpb) and the "
            "oracle is NEVER run at scoring. 0 at or below the baseline recipe, 0.5 at the "
            "demonstrated in-budget ceiling, 1.0 only at speedup = ref_speedup**2."),
        "speedup_semantics": (
            "the `speedup` field carries baseline_bpb / candidate_val_bpb at a FIXED wall-clock "
            "training budget -- a QUALITY ratio (bits-per-byte, lower is better), not a wall-clock "
            "ratio. Nothing in the reward is timed; there is no baseline_ms. reward.md's bounded log "
            "form is applied to that ratio (it is ratio-agnostic; only the variable's name says "
            "latency)."),
        "hard_fail_reasons": reasons,
        "speedup": speedup,
        "ref_speedup": REF_SPEEDUP,
        "baseline_bpb": BASELINE_BPB,
        "candidate_val_bpb": val_bpb,
        "param_floor": PARAM_FLOOR,
        "passed": npass,
        "total": total,
        "tests": {"passed": npass, "total": total},
        "tests_expected_total": EXPECTED_TOTAL_CHECKS,
        "pass_rate": (npass / total if total else 0.0),
        "cv": reward_json["cv"],
        "metric_kind": "quality_at_fixed_budget",
        "metric_name": "val_bpb",
        "metric_direction": "lower_is_better",
        "timing_measured": False,
        "checks": correctness["checks"],
        "failed_checks": [c for c in correctness["checks"] if not c["passed"]],
    }
    verifier_state = {
        "task_id": "e2e-a3-moe-train-budget",
        "task_type": "performance",
        "reward": reward,
        "hard_fail_reasons": reasons,
        "speedup": speedup,
        "ref_speedup": REF_SPEEDUP,
        "baseline_bpb": BASELINE_BPB,
        "candidate_val_bpb": val_bpb,
        "passed": npass,
        "total": total,
        "expected_total": EXPECTED_TOTAL_CHECKS,
    }
    if extra:
        metrics.update(extra)
        verifier_state["detail"] = {k: v for k, v in extra.items()}

    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "reward.txt": f"{reward:.6f}\n",
        "reward.json": json.dumps(reward_json, indent=2, sort_keys=True, default=str) + "\n",
        "metrics.json": json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        "verifier_state.json": json.dumps(verifier_state, indent=2, sort_keys=True, default=str) + "\n",
        "correctness_results.json": json.dumps(correctness, indent=2, sort_keys=True, default=str) + "\n",
        "benchmark_results.json": json.dumps(benchmark, indent=2, sort_keys=True, default=str) + "\n",
    }
    for name in RESULT_FILES:
        try:
            (VERIFIER_DIR / name).write_text(payloads[name], encoding="utf-8")
        except Exception:
            pass
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.message}")
    print(f"REWARD={reward:.6f} speedup={speedup} ref_speedup={REF_SPEEDUP} "
          f"baseline_bpb={BASELINE_BPB} val_bpb={val_bpb} hard_fail_reasons={reasons}")


# -----------------------------------------------------------------------------
# anti-tamper: the frozen-surface state test.sh recorded around the training run
# -----------------------------------------------------------------------------
def frozen_surface_verdict() -> tuple[list[str], dict[str, Any]]:
    """Read the before/after frozen-surface fingerprint test.sh recorded around the
    solver's training run. Any change to a frozen file (verifier, timer, manifest,
    held-out corpus) is gate 3 + gate 4. A missing record is itself gate 3: the scored
    path always produces one."""
    reasons: list[str] = []
    info: dict[str, Any] = {}
    p = os.environ.get("E2E_FROZEN_STATE", "")
    if not p or not Path(p).is_file():
        return [R_CHEAT], {"frozen_surface": "state record missing (expected from test.sh)"}
    try:
        st = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as exc:
        return [R_CHEAT], {"frozen_surface": f"state record unreadable: {type(exc).__name__}: {exc}"}
    changed = list(st.get("changed") or [])
    mode_changed = list(st.get("mode_changed") or [])
    unexpected = list(st.get("unexpected_files") or [])
    info["frozen_surface"] = {"changed": changed, "mode_changed": mode_changed,
                              "unexpected_files": unexpected,
                              "files_fingerprinted": st.get("n_files")}
    if changed:
        reasons += [R_CHEAT, R_FORBIDDEN]
    if mode_changed or unexpected:
        reasons += [R_CHEAT]
    obs = os.environ.get("E2E_OBSERVED_TRAIN_UID", "")
    req = os.environ.get("E2E_REQUESTED_TRAIN_UID", "")
    info["training_privileges"] = {
        "requested_uid": req or None,
        "observed_euid": obs or None,
        "dropped": bool(req) and obs == req,
        "note": ("the candidate's training ran under this euid; chmod 0000 on /tests only bites "
                 "for a non-root uid, so `dropped: false` means the held-out quarantine was "
                 "nominal for that run"),
    }
    if req and obs != req:
        # not a cheat by the candidate, but the run's quarantine was weaker than designed:
        # surface it loudly instead of silently scoring as if it had held.
        info["training_privileges"]["WARNING"] = (
            f"requested uid {req} but training ran as euid {obs or 'unknown'}")
    if st.get("overrides_seen"):
        info["frozen_surface"]["overrides_seen"] = st["overrides_seen"]
        reasons += [R_CHEAT]
    return reasons, info


# -----------------------------------------------------------------------------
# model loading / eval  (machinery lifted verbatim from parameter-golf)
# -----------------------------------------------------------------------------
def sanitize_python_path(*extra_blocked: Path) -> None:
    blocked = {Path.cwd().resolve()} | {p.resolve() for p in extra_blocked}
    clean = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            r = Path(entry).resolve()
        except OSError:
            continue
        if r in blocked:
            continue
        clean.append(entry)
    sys.path[:] = clean


def import_solution(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_train_gpt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_train_gpt"] = module
    ws = str(path.parent.resolve())
    inserted = ws not in sys.path
    if inserted:
        sys.path.insert(0, ws)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(ws)
            except ValueError:
                pass
    return module


def call_model_loader(module: Any, ckpt: Path, device):
    import torch
    args = module.Hyperparameters() if hasattr(module, "Hyperparameters") else None
    loader = getattr(module, "load_model_for_verification", None)
    if loader is None:
        raise RuntimeError("train_gpt.py must define load_model_for_verification(checkpoint_path, device)")
    sig = inspect.signature(loader)
    kwargs, positional = {}, []
    for name, p in sig.parameters.items():
        if name in {"checkpoint_path", "path", "model_path"}:
            kwargs[name] = ckpt
        elif name == "device":
            kwargs[name] = device
        elif p.default is inspect._empty:
            if not positional:
                positional.append(ckpt)
            elif len(positional) == 1:
                positional.append(device)
    loaded = loader(*positional, **kwargs)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        loaded_args, model = loaded
        if loaded_args is not None:
            args = loaded_args
    else:
        model = loaded
    if not isinstance(model, torch.nn.Module):
        raise TypeError("load_model_for_verification must return an nn.Module or (args, model)")
    return args, model.to(device).eval()


def build_sentencepiece_luts(sp, vocab_size, device):
    import numpy as np
    import torch
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int32)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int32, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))


def _resolve_tokenizer(args) -> Path:
    cand = Path(str(getattr(args, "tokenizer_path", DEFAULT_TOKENIZER_PATH)))
    if cand.exists():
        return cand
    if DEFAULT_TOKENIZER_PATH.exists():
        return DEFAULT_TOKENIZER_PATH
    raise FileNotFoundError(f"No tokenizer at {cand} or {DEFAULT_TOKENIZER_PATH}")


def hidden_validation_tokens(args, device):
    import numpy as np
    import sentencepiece as spm
    import torch
    if not HOLDOUT_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Held-out corpus not found at {HOLDOUT_CORPUS_PATH}")
    sp = spm.SentencePieceProcessor(model_file=str(_resolve_tokenizer(args)))
    raw = HOLDOUT_CORPUS_PATH.read_text(encoding="utf-8")
    text = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("# provenance-marker"))
    ids = np.array(sp.encode(text), dtype=np.int64)
    seq_len = int(getattr(args, "train_seq_len", DEFAULT_SEQ_LEN) or DEFAULT_SEQ_LEN)
    if ids.size <= seq_len + 1:
        ids = np.tile(ids, (seq_len + 2) // max(ids.size, 1) + 1)
    usable = ((ids.size - 1) // seq_len) * seq_len
    val_tokens = torch.from_numpy(ids[: usable + 1]).contiguous()
    vocab_size = int(getattr(args, "vocab_size", sp.vocab_size()) or sp.vocab_size())
    return val_tokens, build_sentencepiece_luts(sp, vocab_size, device), sp


def _eval_val_one_pass(model, val_tokens, luts, device, seq_len, val_batch_tokens, offset):
    import torch
    import torch.nn.functional as F
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts
    val_batch_tokens = max(val_batch_tokens, seq_len)
    batch_seqs = max(val_batch_tokens // seq_len, 1)
    n = val_tokens.numel()
    if offset < 0 or offset >= n - 1:
        raise ValueError("offset out of range")
    usable = ((n - offset - 1) // seq_len) * seq_len
    total_seqs = usable // seq_len
    if total_seqs <= 0:
        raise ValueError("corpus too short")
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    for seq_start in range(0, total_seqs, batch_seqs):
        seq_end = min(seq_start + batch_seqs, total_seqs)
        raw_start = offset + seq_start * seq_len
        raw_end = offset + seq_end * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        try:
            out = model(x, y)
        except Exception:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = model(x, y)
        if isinstance(out, torch.Tensor) and out.ndim >= 1 and out.numel() != 1:
            logits = out.float().reshape(-1, out.shape[-1])
            batch_loss = F.cross_entropy(logits, y.reshape(-1), reduction="mean")
        else:
            batch_loss = out
        btc = float(y.numel())
        val_loss_sum += batch_loss.detach().to(torch.float64) * btc
        val_token_count += btc
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int32)
        token_bytes = token_bytes + (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int32)
        val_byte_count += token_bytes.to(torch.float64).sum()
    val_loss = (val_loss_sum / val_token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def eval_val(model, val_tokens, luts, device, seq_len, val_batch_tokens):
    import torch
    model.eval()
    offsets = (0, seq_len // 3, (2 * seq_len) // 3)
    losses, bpbs = [], []
    with torch.inference_mode():
        for off in offsets:
            try:
                loss, bpb = _eval_val_one_pass(model, val_tokens, luts, device, seq_len, val_batch_tokens, off)
            except ValueError:
                continue
            losses.append(loss)
            bpbs.append(bpb)
    if not bpbs:
        raise RuntimeError("corpus too short for any shifted pass")
    passes = sorted(bpbs)
    losses_sorted = sorted(losses)
    mid = len(passes) // 2
    return float(losses_sorted[mid]), float(passes[mid]), passes


def _probe_shapes(args):
    seq_len = int(getattr(args, "train_seq_len", DEFAULT_SEQ_LEN) or DEFAULT_SEQ_LEN)
    vocab_size = int(getattr(args, "vocab_size", 0) or 0) or 8192
    return max(seq_len, 16), vocab_size


def _logits_and_loss_probe(model, args, device):
    import torch
    import torch.nn.functional as F
    seq_len, vocab_size = _probe_shapes(args)
    arange = torch.arange(seq_len, device=device, dtype=torch.int64)
    x = (arange % vocab_size).reshape(1, -1)
    y = ((arange + 1) % vocab_size).reshape(1, -1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = model(x)
        loss_xy = model(x, y)
    if not isinstance(logits, torch.Tensor):
        raise AssertionError(f"forward(x) returned {type(logits).__name__}, expected logits")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != seq_len:
        raise AssertionError(f"forward(x) shape {tuple(logits.shape)}, expected (1,{seq_len},V)")
    if not logits.dtype.is_floating_point:
        raise AssertionError(f"forward(x) dtype {logits.dtype} not floating-point")
    lf = logits.float()
    pos_var = float(lf.var(dim=1).mean().item())
    if not math.isfinite(pos_var) or pos_var <= LOGITS_POSITION_VARIANCE_FLOOR:
        raise AssertionError(f"forward(x) logits constant across positions (var={pos_var:.2e})")
    V = logits.shape[-1]
    ref = float(F.cross_entropy(lf.reshape(-1, V), y.reshape(-1).clamp_max(V - 1), reduction="mean").item())
    lxy = float(loss_xy.detach().to(torch.float64).item()) if isinstance(loss_xy, torch.Tensor) else float(loss_xy)
    if not math.isfinite(lxy) or not math.isfinite(ref):
        raise AssertionError(f"non-finite probe losses xy={lxy} ce={ref}")
    if abs(lxy - ref) > LOGITS_LOSS_AGREEMENT_TOL:
        raise AssertionError(f"forward(x,y)={lxy:.4f} disagrees with CE(forward(x),y)={ref:.4f}")
    return lxy, ref, pos_var, V


def check_validation(workspace: Path):
    """Correctness / anti-spoof suite. Returns (checks, val_bpb, bpb_passes)."""
    import torch
    checks: list[Check] = []
    if not torch.cuda.is_available():
        return [result("CUDA available", False, "CUDA required", gate=R_BUILD)], None, None
    if not CKPT_PATH.exists():
        return [result("Checkpoint present", False, f"no checkpoint at {CKPT_PATH}", gate=R_BUILD)], None, None
    try:
        module = import_solution(SUBMISSION_DIR / "train_gpt.py")
    except Exception as exc:
        return [result("Import submission", False, f"{type(exc).__name__}: {exc}", gate=R_BUILD)], None, None
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    try:
        args, model = call_model_loader(module, CKPT_PATH, device)
    except Exception as exc:
        return [result("load_model_for_verification", False, f"{type(exc).__name__}: {exc}", gate=R_BUILD)], None, None
    checks.append(result("Import + checkpoint reload", True, "loader returned an nn.Module", gate=R_BUILD))

    n_params = sum(int(p.numel()) for p in model.parameters())
    checks.append(result("Model parameter count", n_params >= MIN_PARAM_COUNT,
                         f"{n_params:,} params (real-LM floor {MIN_PARAM_COUNT:,})", n_params=n_params))
    if n_params < MIN_PARAM_COUNT:
        return checks, None, None

    # A3 CAPACITY FLOOR — harness-recounted total params, weight-tie deduplicated.
    seen, unique_params = set(), 0
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        unique_params += int(p.numel())
    checks.append(result("A3 capacity floor (total params >= floor)", unique_params >= PARAM_FLOOR,
                         f"{unique_params:,} unique params (floor {PARAM_FLOOR:,})",
                         total_params=unique_params, param_floor=PARAM_FLOOR))
    if unique_params < PARAM_FLOOR:
        return checks, None, None

    try:
        loss_ar, ce_ar, pos_var, V = _logits_and_loss_probe(model, args, device)
    except Exception as exc:
        checks.append(result("forward(x)/forward(x,y) CE agreement", False, f"{type(exc).__name__}: {exc}"))
        return checks, None, None
    checks.append(result("forward(x)/forward(x,y) CE agreement", True,
                         f"loss_xy={loss_ar:.4f} ce={ce_ar:.4f} pos_var={pos_var:.2e}",
                         loss_xy=loss_ar, ce_loss=ce_ar, position_variance=pos_var))

    try:
        val_tokens, luts, sp = hidden_validation_tokens(args, device)
    except Exception as exc:
        checks.append(result("Held-out corpus available", False, f"{type(exc).__name__}: {exc}", gate=R_BUILD))
        return checks, None, None
    seq_len, _ = _probe_shapes(args)
    vbt = int(getattr(args, "val_batch_size", DEFAULT_VAL_BATCH_TOKENS) or DEFAULT_VAL_BATCH_TOKENS)
    try:
        val_loss, val_bpb, bpb_passes = eval_val(model, val_tokens, luts, device, seq_len, vbt)
    except Exception as exc:
        checks.append(result("Held-out evaluation", False, f"{type(exc).__name__}: {exc}"))
        return checks, None, None

    # Plausibility band. The LOWER edge is an anti-spoof / anti-contamination gate: no
    # honest recipe trained inside this budget can reach it, so a value below it means a
    # forged loss or held-out contamination, not a better model.
    plausible = math.isfinite(val_bpb) and MIN_PLAUSIBLE_VAL_BPB <= val_bpb <= MAX_PLAUSIBLE_VAL_BPB
    checks.append(result("Held-out val_bpb plausible (anti-spoof band)", plausible,
                         f"val_bpb={val_bpb:.4f} in [{MIN_PLAUSIBLE_VAL_BPB},{MAX_PLAUSIBLE_VAL_BPB}]",
                         gate=(R_CHEAT if (math.isfinite(val_bpb) and val_bpb < MIN_PLAUSIBLE_VAL_BPB) else R_CORRECT),
                         val_bpb=val_bpb, val_loss=val_loss, shift_passes=bpb_passes))

    if val_tokens.numel() >= seq_len + 1:
        import torch.nn.functional as F
        rx = val_tokens[:seq_len].reshape(1, -1).to(device=device, dtype=torch.int64).clamp_max(V - 1)
        ry = val_tokens[1:seq_len + 1].reshape(1, -1).to(device=device, dtype=torch.int64).clamp_max(V - 1)
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = model(rx, ry)
            if isinstance(out, torch.Tensor) and out.numel() != 1:
                loss_real = float(F.cross_entropy(out.float().reshape(-1, out.shape[-1]), ry.reshape(-1), reduction="mean").item())
            else:
                loss_real = float(out.detach().to(torch.float64).item())
        except Exception as exc:
            checks.append(result("OOD: loss(real) < ratio*loss(arange)", False, f"{type(exc).__name__}: {exc}"))
            return checks, val_bpb, bpb_passes
        thr = OOD_LOSS_RATIO * loss_ar
        checks.append(result("OOD: loss(real) < ratio*loss(arange)",
                             math.isfinite(loss_real) and loss_real < thr,
                             f"loss(real)={loss_real:.4f} vs {OOD_LOSS_RATIO}*loss(arange)={thr:.4f}",
                             loss_real=loss_real, loss_arange=loss_ar))
    return checks, val_bpb, bpb_passes


# -----------------------------------------------------------------------------
# reward.md performance pre-gates + bounded reward
# -----------------------------------------------------------------------------
def score(checks: list[Check], val_bpb, bpb_passes, pre_reasons: list[str], extra: dict[str, Any]):
    reasons = list(pre_reasons)

    # gate 6 — reference solution invalid / uncalibrated. Fail CLOSED: never default the
    # constant, never silently treat a missing anchor as 1.0.
    ref_ok = isinstance(REF_SPEEDUP, (int, float)) and math.isfinite(float(REF_SPEEDUP)) and float(REF_SPEEDUP) > 1.0
    base_ok = isinstance(BASELINE_BPB, (int, float)) and math.isfinite(float(BASELINE_BPB)) and float(BASELINE_BPB) > 0
    if not ref_ok or not base_ok:
        reasons.append(R_BAD_REF)
        extra["ref_speedup_status"] = (
            "ANCHOR_NOT_CALIBRATED: reward_model.baseline_bpb / reward_model.ref_speedup absent or <= 1 "
            "in the frozen manifest")
    elif float(REF_SPEEDUP) < REF_SPEEDUP_AUTHORING_FLOOR:
        reasons.append(R_BAD_REF)
        extra["ref_speedup_status"] = (
            f"ref_speedup={float(REF_SPEEDUP):.6f} < authoring floor {REF_SPEEDUP_AUTHORING_FLOOR} "
            "(reward.md legality): task must be reclassified, not scored")

    # gate 1/2/3 — build/import, correctness suite, anti-cheat. Each failing check contributes
    # its own named gate reason.
    for c in checks:
        if not c.passed:
            reasons.append(c.gate)
    if not checks:
        reasons.append(R_BUILD)
    # a TRUNCATED suite (early return) can never score above 0 — the full suite must be
    # present as well as green.
    if len(checks) < EXPECTED_TOTAL_CHECKS:
        reasons.append(R_INCOMPLETE)

    speedup = None
    if base_ok and isinstance(val_bpb, (int, float)) and math.isfinite(val_bpb) and val_bpb > 0:
        speedup = float(BASELINE_BPB) / float(val_bpb)

    # gate 5 — did not beat the baseline recipe it started from
    if speedup is None:
        reasons.append(R_NO_GAIN)
    elif speedup <= 1.0:
        reasons.append(R_NO_GAIN)

    if reasons:
        emit(0.0, reasons, checks, speedup, val_bpb, bpb_passes, extra)
        return 0.0, reasons

    reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(float(REF_SPEEDUP)) - 1.0)))
    reward = min(1.0, max(0.0, reward))
    emit(reward, [], checks, speedup, val_bpb, bpb_passes, extra)
    return reward, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-zero", metavar="REASON", default=None,
                    help="named reason from the shared vocabulary; emits the full output contract "
                         "with reward 0")
    ap.add_argument("--detail", default="", help="human-readable detail for --emit-zero")
    ns, _ = ap.parse_known_args()
    if ns.emit_zero:
        known = {R_BUILD, R_CORRECT, R_CHEAT, R_FORBIDDEN, R_NO_GAIN, R_BAD_REF,
                 R_CRASH, R_UNNAMED, R_INCOMPLETE}
        reason = ns.emit_zero.strip()
        if reason not in known:
            reason = R_BUILD
        detail = ns.detail or ns.emit_zero
        emit(0.0, [reason], [result("Harness pre-flight", False, detail, gate=reason)],
             extra={"harness_message": detail})
        return 1

    extra: dict[str, Any] = {}
    pre_reasons: list[str] = []
    if _MAN_PROBLEMS:
        pre_reasons.append(R_BUILD)
        extra["manifest_problems"] = _MAN_PROBLEMS
    # Refuse to score against an in-image (solver-reachable) evaluation surface: after the
    # de-leak rebuild there is no /opt/verifier at all, so its presence means a stale image.
    if Path("/opt/verifier").exists():
        pre_reasons.append(R_FORBIDDEN)
        extra["stale_baked_verifier"] = (
            "/opt/verifier exists in this image — the frozen evaluation surface must be uploaded "
            "fresh at /tests only; refusing to score against a solver-reachable copy")
    tamper_reasons, tamper_info = frozen_surface_verdict()
    pre_reasons += tamper_reasons
    extra.update(tamper_info)

    sanitize_python_path(SUBMISSION_DIR)
    try:
        checks, val_bpb, bpb_passes = check_validation(SUBMISSION_DIR)
    except Exception as exc:
        checks = [result("Verifier error", False, f"{type(exc).__name__}: {exc}", gate=R_BUILD)]
        val_bpb, bpb_passes = None, None
    reward, reasons = score(checks, val_bpb, bpb_passes, pre_reasons, extra)
    return 0 if (reward > 0.0 and not reasons) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:                      # never exit without the output contract
        try:
            emit(0.0, [R_CRASH], [result("Verifier crashed", False, f"{type(exc).__name__}: {exc}", gate=R_CRASH)])
        finally:
            raise SystemExit(1)
