"""e2e-a4-token-efficiency-budget verifier (family A; A4 优化器/sample-efficiency).

THE BUDGET IS *TRAINING TOKENS*, NOT WALL-CLOCK -- the axis that makes this task distinct
from every wall-clock sibling in this lane. Because the budget is tokens, every THROUGHPUT
lever (fp8 / torch.compile / bigger batch / faster loader) is worth EXACTLY ZERO, so the only
way to a lower held-out loss is real SAMPLE EFFICIENCY (optimizer / architecture / data
schedule) -- the modded-nanogpt record axis.

REWARD MODEL (the bench reward spec — performance class, BOUNDED log form):

    speedup     = baseline_bpb / median_val_bpb            # quality ratio at a FIXED budget
    ref_speedup = baseline_bpb / oracle_val_bpb            # FROZEN authoring constant
    reward      = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0          in [0, 1]

`baseline_bpb` and `ref_speedup` are **frozen constants** carried in the frozen manifest
(`reward_model` block) and calibrated on NVIDIA H20 at authoring time. This scorer NEVER runs the
oracle and NEVER runs the baseline: it reads the two constants and trains only the candidate.
There is no oracle/reference recipe anywhere in the task image.

Semantics: `speedup == ref_speedup => 0.0 (must EXCEED it to score) ; `speedup >= ref_speedup**2` => capped 1.0 ;
`speedup <= 1` (did not beat the tuned-AdamW recipe it started from) => 0 via pre-gate 5.
Matching the *baseline* therefore scores **0**, not 1.0 — the old open-ended reading
("matching it scores 1.0, no upper cap") is retired.

The metric is a *quality* metric at a fixed training budget, not a latency measurement: no
timing enters the reward (`timing_measured: false`). `"speedup"` is kept as the field name for
schema compatibility and is described by `metric_kind` / `metric_name` / `metric_direction`.

WHAT THE HARNESS DOES (nothing is trusted from the candidate):
  * builds a per-seed BUDGETED SHARD of exactly `token_budget` tokens from the digest-pinned
    corpus and QUARANTINES the full corpus, so the token budget holds BY CONSTRUCTION -- the
    budget is never a number the candidate reports;
  * QUARANTINES the frozen surface (/tests) for the duration of every training call and
    sha256-fingerprints it before/after, so the held-out corpus and the calibrated constants
    are unreadable while the candidate's code runs and any change is detected;
  * runs the candidate's `train_for_budget(seed, out_dir, budget_seconds)` from an ISOLATED
    staging dir with every *.pt/*.safetensors/>8 MB blob stripped (blocks the smuggled
    checkpoint), under an EXTERNAL hard kill (setsid pgroup -> SIGTERM -> SIGKILL) on a
    generous wall-clock FEASIBILITY GUARD (not the budget);
  * reloads each seed's checkpoint in a clean process and re-counts parameters against a hard
    cap so "just scale the model up" is not the answer;
  * requires >=3 distinct checkpoints across seeds (byte-identical => smuggled/cached => 0);
  * runs the parameter-golf anti-spoof kit and scores held-out val_bpb on the HIDDEN
    WikiText-103 TEST split, median over seeds.

The eval / anti-spoof blocks are lifted VERBATIM from e2e-subquad-lm-golf via
e2e-subquad-lm-30min; the sub-quadratic gate is replaced by this task's parameter cap.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Frozen paths + manifest-driven thresholds
# -----------------------------------------------------------------------------
WORKSPACE = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))

# The frozen surface lives at /tests ONLY. There is deliberately NO /opt/verifier fallback:
# a baked fallback copy is (a) a stale-scoring hazard and (b) an in-image leak of the held-out
# corpus and the calibrated constants (both were live in the :v2 image).
FROZEN_DIR = Path("/tests")


def _frozen(name: str, env: str | None = None) -> Path:
    if env and os.environ.get(env):
        return Path(os.environ[env])
    return FROZEN_DIR / name


MANIFEST_PATH = _frozen("verifier-correctness-manifest.json", "E2E_MANIFEST_PATH")
HOLDOUT_CORPUS_PATH = _frozen("holdout_corpus.txt", "E2E_HOLDOUT_PATH")
DEFAULT_TOKENIZER_PATH = Path("/data/tokenizers/wikitext_bpe8k.model")

VERIFIER_DIR = Path("/logs/verifier")
# The verifier output contract ("5-file contract" — 6 files in the corrected bench-wide form,
# identical to the embed-compress-golf sibling). Emitted on EVERY exit path.
RESULT_FILES = (
    "reward.txt",
    "reward.json",
    "metrics.json",
    "verifier_state.json",
    "correctness_results.json",
    "benchmark_results.json",
)

# ---- named hard-fail reasons (reward.md performance pre-gates 1..6) ----
R_BUILD = "build_or_entry_contract_failed"          # gate 1
R_CORRECT = "correctness_failed"                    # gate 2
R_CHEAT = "anti_cheat_triggered"                    # gate 3
R_FORBIDDEN = "forbidden_edit_path"                 # gate 4
R_NO_GAIN = "speedup_not_above_baseline"            # gate 5
R_BAD_REF = "ref_speedup_invalid_or_missing"        # gate 6
R_CRASH = "verifier_crashed"
R_UNNAMED = "zero_without_named_reason"
R_INCOMPLETE = "gate_suite_incomplete"

REF_SPEEDUP_AUTHORING_FLOOR = 1.15   # reward.md authoring legality floor

DEFAULT_SEQ_LEN = 1024
DEFAULT_VAL_BATCH_TOKENS = 32_768

_DEFAULTS = {
    # ---- THE BUDGET IS *TOKENS*, NOT WALL-CLOCK (this task's distinguishing axis) ----
    # Enforced BY CONSTRUCTION: the harness materialises a budgeted shard holding EXACTLY
    # `token_budget` tokens and quarantines the full corpus for the scored run, so the recipe
    # physically cannot consume more tokens than the budget. Nothing is trusted from the
    # candidate; no in-loop token counter is believed.
    "token_budget": 24_000_000,
    # wall-clock is only a FEASIBILITY GUARD so a slower-but-smarter recipe is never penalised
    "budget_seconds": 900.0,             # = MAX_WALLCLOCK_SEC (guard, NOT the budget)
    "budget_grace_seconds": 60.0,
    "n_seeds": 3,
    "seeds": [1337, 2027, 90101],
    "min_plausible_val_bpb": 1.00,       # anti-forged-loss / anti-contamination floor
    "quality_floor_val_bpb": 3.00,       # the low floor (any real LM clears)
    "min_param_count": 1_000_000,
    # ---- PARAMETER CAP (harness re-counts; stops "just scale the model up") ----
    "max_param_count": 45_000_000,
    "min_param_count_floor": 20_000_000,
    # anti-spoof constants (parameter-golf, verbatim)
    "ood_loss_ratio": 0.7,
    "logits_position_variance_floor": 1e-4,
    "logits_loss_agreement_tol": 5e-2,
    "training_uid": 65534,               # run the candidate's training as nobody (see below)
    "expected_total_checks": None,        # derived: 1 entry + n_seeds*7 + divergence + aggregate
}

# ---- FROZEN reward constants (read from the manifest's reward_model block; never guessed) ----
_REWARD_KEYS = ("baseline_bpb", "ref_speedup", "baseline_cv")


def load_manifest() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    cfg["_manifest_problems"] = []
    cfg["baseline_bpb"] = None
    cfg["ref_speedup"] = None
    cfg["baseline_cv"] = None
    if not MANIFEST_PATH.exists():
        cfg["_manifest_problems"].append(f"frozen manifest missing at {MANIFEST_PATH}")
        return cfg
    try:
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        cfg["_manifest_problems"].append(f"frozen manifest unreadable: {type(exc).__name__}: {exc}")
        return cfg
    th = m.get("thresholds", m)
    for k in _DEFAULTS:
        if k in th and th[k] is not None:
            cfg[k] = th[k]
    qg = m.get("quality_gate") or {}
    for k in _DEFAULTS:
        if k in qg and qg[k] is not None:
            cfg[k] = qg[k]
    bud = m.get("budget") or {}
    if bud.get("token_budget") is not None:
        cfg["token_budget"] = bud["token_budget"]
    if bud.get("wallclock_guard_seconds") is not None:
        cfg["budget_seconds"] = bud["wallclock_guard_seconds"]
    if bud.get("budget_grace_seconds") is not None:
        cfg["budget_grace_seconds"] = bud["budget_grace_seconds"]
    if m.get("n_seeds") is not None:
        cfg["n_seeds"] = m["n_seeds"]
    if m.get("seeds"):
        cfg["seeds"] = m["seeds"]
    at = m.get("anti_tamper") or {}
    if "training_uid" in at:
        cfg["training_uid"] = at["training_uid"]      # may be null to disable the drop
    rm = m.get("reward_model") or {}
    for k in _REWARD_KEYS:
        cfg[k] = rm.get(k)
    # 🔴 NO ENVIRONMENT OVERRIDES on the scored path. The previous version honoured
    # E2E_BUDGET_SECONDS_OVERRIDE / E2E_TOKEN_BUDGET_OVERRIDE / E2E_N_SEEDS_OVERRIDE and
    # E2E_STRONG_BASELINE_BPB_OVERRIDE -- i.e. the reward's own anchor was settable from the
    # environment. The container's entrypoint is a LOGIN shell, which sources ~/.bashrc, a file
    # the solver owns, so those overrides were a live reward-inflation vector. They are gone.
    # The author's calibration driver sets the constants in the frozen manifest instead.
    # 1 entry contract + per seed {checkpoint, param count, CE agreement, param cap,
    # quality floor, plausibility, OOD} + cross-seed divergence + aggregate.
    # MEASURED: 3 seeds => 24 checks, matching the pre-modification suite size exactly
    # (no test case was removed).
    cfg["expected_total_checks"] = 1 + int(cfg["n_seeds"]) * 7 + 1 + 1
    return cfg


# -----------------------------------------------------------------------------
# Check / result helpers
# -----------------------------------------------------------------------------
class Check:
    def __init__(self, name: str, passed: bool, message: str, details: dict[str, Any] | None = None, hard: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.hard = hard


def result(name: str, passed: bool, message: str, hard: bool = True, **details: Any) -> Check:
    return Check(name, passed, message, details, hard)


def sanitize_python_path(workspace: Path) -> None:
    blocked = {workspace.resolve(), Path.cwd().resolve()}
    clean: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved in blocked:
            continue
        clean.append(entry)
    sys.path[:] = clean


def import_solution(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_train_gpt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_train_gpt"] = module
    workspace_str = str(path.parent.resolve())
    inserted = False
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(workspace_str)
            except ValueError:
                pass
    return module


# -----------------------------------------------------------------------------
# G2: harness-owned wall-clock — run the agent's train_for_budget in an
# EXTERNALLY-killed subprocess (process group killed at budget + grace).
# -----------------------------------------------------------------------------
_TRAIN_RUNNER = r'''
import importlib.util, os, sys
sub = sys.argv[1]; seed = int(sys.argv[2]); out_dir = sys.argv[3]; budget = float(sys.argv[4])
spec = importlib.util.spec_from_file_location("submitted_train_gpt", sub)
mod = importlib.util.module_from_spec(spec); sys.modules["submitted_train_gpt"] = mod
sys.path.insert(0, os.path.dirname(os.path.abspath(sub)))
rep = os.environ.get("E2E_UID_REPORT")
if rep:
    try:
        open(rep, "w").write("%d %d\n" % (os.getuid(), os.geteuid()))
    except Exception:
        pass
spec.loader.exec_module(mod)
fn = getattr(mod, "train_for_budget", None)
if fn is None:
    print("train_for_budget missing", file=sys.stderr); sys.exit(7)
# DATA_PATH in env already points at the harness-built BUDGETED SHARD (exactly
# token_budget tokens); the full corpus is quarantined for the duration of this run, so
# the token budget holds BY CONSTRUCTION rather than by trusting a counter.
fn(seed=seed, out_dir=out_dir, budget_seconds=budget)
'''


# -----------------------------------------------------------------------------
# THE TOKEN BUDGET (G2, harness-owned, enforced BY CONSTRUCTION)
# -----------------------------------------------------------------------------
# A training-token budget cannot be enforced by asking the candidate how many tokens it
# used (that is a self-report -- G2 forbids it), and counting reads of a memmap from
# outside is unreliable. So the harness makes over-spending PHYSICALLY IMPOSSIBLE:
#   1. it slices EXACTLY `token_budget` tokens out of the digest-pinned corpus into a
#      per-seed budgeted shard, and points DATA_PATH at that shard;
#   2. it QUARANTINES the full baked corpus (chmod 000 as root) for the duration of the
#      scored run, so the shard is the only readable token source.
# A recipe may still read its shard in any order / packing / curriculum it likes (the
# data-order lever is preserved, G6: the loader impl is open, its byte content is pinned).
_CORPUS_QUARANTINE_MODE = 0o000

# ---- frozen-surface anti-tamper -------------------------------------------------------------
# The candidate's training entry runs as uid 0 in this container, so /tests cannot be made
# cryptographically unreachable. What the harness CAN do is (a) make the held-out corpus and the
# calibrated constants unreadable for the duration of every training call and (b) DETECT any
# change to them. Both a content change and a mode change are red lines.
_TAMPER: dict[str, Any] = {"changed": [], "mode_changed": [], "unexpected_files": [], "n_files": 0}
_PRIV_OBSERVED: list[dict[str, Any]] = []


def _fingerprint_frozen() -> dict[str, dict[str, Any]]:
    rec: dict[str, dict[str, Any]] = {}
    if not FROZEN_DIR.is_dir():
        return rec
    for p in sorted(FROZEN_DIR.iterdir()):
        if not p.is_file():
            continue
        h = hashlib.sha256()
        try:
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rec[p.name] = {"sha256": h.hexdigest(), "mode": p.stat().st_mode & 0o7777}
        except Exception:
            rec[p.name] = {"sha256": "unreadable", "mode": -1}
    return rec


def _quarantine_frozen() -> int:
    try:
        saved = FROZEN_DIR.stat().st_mode & 0o7777
        os.chmod(FROZEN_DIR, _CORPUS_QUARANTINE_MODE)
        return saved
    except Exception:
        return -1


def _restore_and_check_frozen(before: dict[str, dict[str, Any]], saved_mode: int) -> None:
    if saved_mode >= 0:
        try:
            os.chmod(FROZEN_DIR, saved_mode)
        except Exception:
            pass
    after = _fingerprint_frozen()
    _TAMPER["n_files"] = len(before)
    for name, rec in before.items():
        cur = after.get(name)
        if cur is None:
            if name not in _TAMPER["changed"]:
                _TAMPER["changed"].append(name)
            continue
        if cur["sha256"] != rec["sha256"] and name not in _TAMPER["changed"]:
            _TAMPER["changed"].append(name)
        if cur["mode"] != rec["mode"] and name not in _TAMPER["mode_changed"]:
            _TAMPER["mode_changed"].append(name)
    for name in after:
        if name not in before and name not in _TAMPER["unexpected_files"]:
            _TAMPER["unexpected_files"].append(name)


def build_budgeted_shard(corpus_dir: Path, shard_dir: Path, token_budget: int, seed: int,
                         cfg: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Materialise a shard of EXACTLY token_budget uint16 tokens. Deterministic per seed."""
    import numpy as _np
    shard_dir.mkdir(parents=True, exist_ok=True)
    src = corpus_dir / "train.bin"
    if not src.exists():
        return False, f"frozen corpus missing at {src}", {}
    d = _np.memmap(src, dtype=_np.uint16, mode="r")
    have = int(d.shape[0])
    if have < token_budget:
        return False, f"corpus has {have} tokens < budget {token_budget}", {"corpus_tokens": have}
    # Deterministic, seed-dependent contiguous window (single-epoch: no repetition, which
    # is what avoids the MEASURED memorisation dead end of a repeated training set).
    span = have - token_budget
    off = 0 if span <= 0 else (hash((seed, token_budget)) % span if span > 1 else 0)
    off = int(abs(off))
    _np.asarray(d[off:off + token_budget], dtype=_np.uint16).tofile(shard_dir / "train.bin")
    # carry the solver's local val monitor + meta through unchanged
    for extra in ("val.bin", "meta.pkl"):
        sp = corpus_dir / extra
        if sp.exists():
            shutil.copy2(sp, shard_dir / extra)
    got = (shard_dir / "train.bin").stat().st_size // 2
    return (got == token_budget), f"shard tokens={got} (budget {token_budget}) offset={off}", {
        "shard_tokens": got, "offset": off, "corpus_tokens": have}


def quarantine_corpus(corpus_dir: Path) -> dict[str, int]:
    """chmod 000 the full corpus so the budgeted shard is the ONLY readable token source.
    Returns the saved modes so the caller can restore them."""
    saved: dict[str, int] = {}
    for pth in (corpus_dir / "train.bin", corpus_dir):
        try:
            saved[str(pth)] = pth.stat().st_mode & 0o777
            os.chmod(pth, _CORPUS_QUARANTINE_MODE)
        except Exception:
            pass
    return saved


def restore_corpus(saved: dict[str, int]) -> None:
    for k, mode in saved.items():
        try:
            os.chmod(k, mode)
        except Exception:
            pass


def check_param_cap(model: torch.nn.Module, cfg: dict[str, Any]) -> Check:
    """G2: the harness RE-COUNTS parameters from the reloaded artifact. The cap stops the
    degenerate 'just scale the model up' answer (at a fixed TOKEN budget a bigger model is
    monotonically better, so without a cap the task would reward buying HBM, not sample
    efficiency); the floor blocks a degenerate tiny model."""
    n = int(sum(p.numel() for p in model.parameters()))
    cap = int(cfg["max_param_count"])
    floor = int(cfg.get("min_param_count_floor", 0))
    if n > cap:
        return result("param_cap", False,
                      f"parameter cap exceeded: harness re-counted {n:,} > cap {cap:,}",
                      n_params=n, cap=cap)
    if n < floor:
        return result("param_cap", False,
                      f"below the parameter floor: {n:,} < {floor:,} (degenerate model)",
                      n_params=n, floor=floor)
    return result("param_cap", True,
                  f"parameters {n:,} within [{floor:,}, {cap:,}]", n_params=n, cap=cap, floor=floor)


def train_one_seed(submission_py: Path, seed: int, out_dir: Path, cfg: dict[str, Any]) -> tuple[bool, float, str]:
    """Run the agent's recipe for `seed`, killing the whole process group at
    budget + grace. Returns (checkpoint_exists, wall_seconds, note).

    ANTI-BAKE ISOLATION (R2): the recipe is run from an ISOLATED staging dir that
    holds a copy of the submission's *source* files but NONE of its pre-existing
    weight/data artifacts, and /app/submission is NOT on sys.path or cwd. This
    stops `train_for_budget` from `shutil.copy`-ing a pre-trained checkpoint it
    baked into the submission dir (which would defeat the wall-clock premise).
    A legitimate multi-file recipe still works — its .py/.json helpers are copied;
    only model-weight blobs are withheld."""
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = float(cfg["budget_seconds"])
    grace = float(cfg["budget_grace_seconds"])
    # --- build the isolated staging dir (source only; strip weight/data blobs) ---
    stage = out_dir / "_stage_src"
    stage.mkdir(parents=True, exist_ok=True)
    _WEIGHT_EXT = {".pt", ".pth", ".ptz", ".bin", ".safetensors", ".ckpt", ".npz",
                   ".npy", ".pkl", ".pickle", ".h5", ".onnx", ".gguf", ".pt2"}
    src_root = submission_py.parent
    try:
        for p in src_root.rglob("*"):
            if not p.is_file() or "_stage_src" in p.parts or p.name.startswith("_train_runner"):
                continue
            if p.suffix.lower() in _WEIGHT_EXT or p.stat().st_size > 8 * 1024 * 1024:
                continue  # withhold weight blobs + any suspiciously large file (baked weights)
            rel = p.relative_to(src_root)
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, stage / rel)
    except Exception:
        pass
    staged_py = stage / submission_py.name
    if not staged_py.exists():
        shutil.copy2(submission_py, staged_py)
    runner = stage / "_train_runner.py"
    runner.write_text(_TRAIN_RUNNER, encoding="utf-8")
    # --- build the per-seed BUDGETED SHARD and point DATA_PATH at it, then quarantine
    #     the full corpus so the budget cannot be exceeded (see build_budgeted_shard) ---
    corpus_dir = Path(os.environ.get("E2E_CORPUS_DIR", "/data/datasets/wikitext_bpe8k"))
    shard_dir = out_dir / "_budgeted_shard"
    ok_shard, shard_note, shard_info = build_budgeted_shard(
        corpus_dir, shard_dir, int(cfg["token_budget"]), seed, cfg)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["SEED"] = str(seed)
    env["TOKEN_BUDGET"] = str(int(cfg["token_budget"]))
    env["MAX_WALLCLOCK_SEC"] = str(budget)
    env["MAX_PARAMS"] = str(int(cfg["max_param_count"]))
    env["OUT_CKPT"] = str(out_dir / "model_ckpt.pt")
    if ok_shard:
        env["DATA_PATH"] = str(shard_dir)          # the ONLY readable token source
    py = sys.executable or shutil.which("python3") or "python3"

    # --- PRIVILEGE DROP -------------------------------------------------------------------
    # 🔴 chmod 0000 means NOTHING to uid 0. The full-corpus quarantine (which is what makes the
    # TOKEN budget hold "by construction") and the /tests quarantine (which protects the held-out
    # split) are therefore only nominal while the candidate's training runs as root: it could
    # simply open /data/datasets/wikitext_bpe8k/train.bin and train on the whole corpus, or read
    # the held-out corpus it is scored on. Running the training entry under an unprivileged uid is
    # what turns both quarantines real. Everything the training legitimately needs is opened to
    # that uid first; the corpus and the frozen surface are not.
    train_uid = cfg.get("training_uid")
    preexec = os.setsid
    if train_uid:
        try:
            # the WHOLE path chain must be traversable by the unprivileged uid, not just the
            # leaf: tempfile.mkdtemp() creates a 0700 root-owned parent, so without this the
            # exec fails instantly with "cannot access parent directories" (measured).
            anc = out_dir.resolve()
            while str(anc) not in ("/", str(anc.parent)):
                try:
                    if anc.stat().st_uid == os.getuid():
                        os.chmod(anc, 0o755)
                except Exception:
                    pass
                anc = anc.parent
            os.chmod(stage, 0o777)
            os.chmod(out_dir, 0o777)
            for p in stage.rglob("*"):
                try:
                    os.chmod(p, 0o777 if p.is_dir() else 0o666)
                except Exception:
                    pass
            if ok_shard:
                os.chmod(shard_dir, 0o755)
                for p in shard_dir.iterdir():
                    try:
                        os.chmod(p, 0o644)
                    except Exception:
                        pass
        except Exception:
            pass
        home = out_dir / "_trainhome"
        home.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(home, 0o777)
        except Exception:
            pass
        env["HOME"] = str(home)
        env["TMPDIR"] = str(home)
        env["XDG_CACHE_HOME"] = str(home / ".cache")
        env["TRITON_CACHE_DIR"] = str(home / ".triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(home / ".inductor")
        env["E2E_UID_REPORT"] = str(out_dir / "_runner_uid.txt")
        _uid = int(train_uid)

        def preexec():                       # noqa: F811 - deliberate rebinding
            os.setsid()
            try:
                os.setgroups([])
                os.setgid(_uid)
                os.setuid(_uid)
            except Exception:
                pass                          # recorded below from the child's own report

    saved_modes = quarantine_corpus(corpus_dir) if ok_shard else {}
    # ALSO quarantine + fingerprint the frozen surface for the duration of the training call:
    # the held-out corpus and the calibrated constants must not be readable while the
    # candidate's own code runs, and any change to them must be detected.
    frozen_before = _fingerprint_frozen()
    frozen_saved = _quarantine_frozen()
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [py, str(runner), str(staged_py), str(seed), str(out_dir), str(budget)],
        cwd=str(stage), env=env, preexec_fn=preexec,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    note = "completed"
    try:
        proc.wait(timeout=budget + grace)
    except subprocess.TimeoutExpired:
        note = "hard-killed at budget+grace (expected for a well-tuned recipe that fills the budget)"
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(5)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    wall = time.monotonic() - t0
    # always un-quarantine, even on kill
    _restore_and_check_frozen(frozen_before, frozen_saved)
    restore_corpus(saved_modes)
    ckpt = out_dir / "model_ckpt.pt"
    obs = None
    try:
        obs = (out_dir / "_runner_uid.txt").read_text().split()[1]
    except Exception:
        obs = None
    _PRIV_OBSERVED.append({"seed": seed, "requested_uid": train_uid, "observed_euid": obs,
                           "dropped": bool(train_uid) and obs == str(train_uid)})
    try:
        if ckpt.exists():
            os.chmod(ckpt, 0o644)
    except Exception:
        pass
    if not ok_shard:
        return False, wall, f"HARNESS FAILURE building the budgeted shard: {shard_note}"
    return ckpt.exists(), wall, f"{note}; budget={cfg['token_budget']} tokens ({shard_note})"


# -----------------------------------------------------------------------------
# Checkpoint loading (agent's loader owns the format) — G3 clean-env reload
# -----------------------------------------------------------------------------
def call_model_loader(module: Any, checkpoint_path: Path, device: torch.device) -> tuple[Any, torch.nn.Module]:
    args = module.Hyperparameters() if hasattr(module, "Hyperparameters") else None
    loader = getattr(module, "load_model_for_verification", None)
    if loader is None:
        raise RuntimeError("train_gpt.py must define load_model_for_verification(path, device).")
    signature = inspect.signature(loader)
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for name, parameter in signature.parameters.items():
        if name in {"checkpoint_path", "path", "model_path", "compressed_checkpoint_path"}:
            kwargs[name] = checkpoint_path
        elif name == "device":
            kwargs[name] = device
        elif parameter.default is inspect._empty:
            if not positional:
                positional.append(checkpoint_path)
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


# -----------------------------------------------------------------------------
# Tokenizer + held-out validation tokens (verbatim)
# -----------------------------------------------------------------------------
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int32)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int32, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def _resolve_tokenizer(args: Any) -> Path:
    candidate = Path(str(getattr(args, "tokenizer_path", DEFAULT_TOKENIZER_PATH)))
    if candidate.exists():
        return candidate
    if DEFAULT_TOKENIZER_PATH.exists():
        return DEFAULT_TOKENIZER_PATH
    raise FileNotFoundError(f"No tokenizer found at {candidate} or {DEFAULT_TOKENIZER_PATH}")


def hidden_validation_tokens(args: Any, device: torch.device):
    if not HOLDOUT_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Held-out validation corpus not found at {HOLDOUT_CORPUS_PATH}")
    tokenizer_path = _resolve_tokenizer(args)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    raw = HOLDOUT_CORPUS_PATH.read_text(encoding="utf-8")
    text_lines = [ln for ln in raw.splitlines() if not ln.startswith("# provenance-marker")]
    text = "\n".join(text_lines)
    ids = np.array(sp.encode(text), dtype=np.int64)
    seq_len = int(getattr(args, "train_seq_len", DEFAULT_SEQ_LEN) or DEFAULT_SEQ_LEN)
    if ids.size <= seq_len + 1:
        ids = np.tile(ids, (seq_len + 2) // max(ids.size, 1) + 1)
    usable = ((ids.size - 1) // seq_len) * seq_len
    val_tokens = torch.from_numpy(ids[: usable + 1]).contiguous()
    vocab_size = int(getattr(args, "vocab_size", sp.vocab_size()) or sp.vocab_size())
    luts = build_sentencepiece_luts(sp, vocab_size, device)
    return val_tokens, luts, sp


# -----------------------------------------------------------------------------
# Evaluation: cross-entropy + bits-per-byte (verbatim)
# -----------------------------------------------------------------------------
def _eval_val_one_pass(model, val_tokens, luts, device, seq_len, val_batch_tokens, offset):
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts
    val_batch_tokens = max(val_batch_tokens, seq_len)
    batch_seqs = max(val_batch_tokens // seq_len, 1)
    n = val_tokens.numel()
    if offset < 0 or offset >= n - 1:
        raise ValueError(f"offset={offset} out of range for {n}-token validation set")
    usable = ((n - offset - 1) // seq_len) * seq_len
    total_seqs = usable // seq_len
    if total_seqs <= 0:
        raise ValueError(f"Validation corpus too short for seq_len={seq_len} at offset={offset}")
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
        batch_token_count = float(y.numel())
        val_loss_sum += batch_loss.detach().to(torch.float64) * batch_token_count
        val_token_count += batch_token_count
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
    model.eval()
    offsets = (0, seq_len // 3, (2 * seq_len) // 3)
    losses: list[float] = []
    bpbs: list[float] = []
    with torch.inference_mode():
        for off in offsets:
            try:
                loss, bpb = _eval_val_one_pass(model, val_tokens, luts, device, seq_len, val_batch_tokens, off)
            except ValueError:
                continue
            losses.append(loss)
            bpbs.append(bpb)
    if not bpbs:
        raise RuntimeError("Validation corpus too short for any shifted pass")
    losses.sort()
    bpbs.sort()
    mid = len(bpbs) // 2
    return float(losses[mid]), float(bpbs[mid])


# -----------------------------------------------------------------------------
# Anti-spoof probes (verbatim)
# -----------------------------------------------------------------------------
def _probe_shapes(args: Any) -> tuple[int, int]:
    seq_len = int(getattr(args, "train_seq_len", DEFAULT_SEQ_LEN) or DEFAULT_SEQ_LEN)
    vocab_size = int(getattr(args, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        vocab_size = 8192
    return max(seq_len, 16), vocab_size


def _logits_and_loss_probe(model, args, device, cfg):
    seq_len, vocab_size = _probe_shapes(args)
    arange = torch.arange(seq_len, device=device, dtype=torch.int64)
    x_arange = (arange % vocab_size).reshape(1, -1)
    y_arange = ((arange + 1) % vocab_size).reshape(1, -1)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x_arange)
            loss_xy = model(x_arange, y_arange)
    if not isinstance(logits, torch.Tensor):
        raise AssertionError(f"forward(x) returned {type(logits).__name__}, expected logits tensor")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != seq_len:
        raise AssertionError(f"forward(x) returned shape {tuple(logits.shape)}, expected (1, {seq_len}, vocab)")
    if not logits.dtype.is_floating_point:
        raise AssertionError(f"forward(x) returned dtype {logits.dtype}, expected floating-point")
    logits_f32 = logits.float()
    pos_var = float(logits_f32.var(dim=1).mean().item())
    if not math.isfinite(pos_var) or pos_var <= float(cfg["logits_position_variance_floor"]):
        raise AssertionError(f"forward(x) logits are constant across positions (var={pos_var:.2e})")
    real_vocab = logits.shape[-1]
    ref_loss = float(F.cross_entropy(
        logits_f32.reshape(-1, real_vocab),
        y_arange.reshape(-1).clamp_max(real_vocab - 1),
        reduction="mean",
    ).item())
    loss_xy_f = float(loss_xy.detach().to(torch.float64).item()) if isinstance(loss_xy, torch.Tensor) else float(loss_xy)
    if not math.isfinite(loss_xy_f) or not math.isfinite(ref_loss):
        raise AssertionError(f"non-finite probe losses: forward(x,y)={loss_xy_f}, CE={ref_loss}")
    if abs(loss_xy_f - ref_loss) > float(cfg["logits_loss_agreement_tol"]):
        raise AssertionError(
            f"forward(x,y)={loss_xy_f:.4f} disagrees with CE(forward(x),y)={ref_loss:.4f} "
            f"by {abs(loss_xy_f - ref_loss):.4f} > {cfg['logits_loss_agreement_tol']}"
        )
    return loss_xy_f, ref_loss, pos_var, real_vocab


def eval_one_checkpoint(module, checkpoint_path: Path, cfg: dict[str, Any]) -> tuple[list[Check], float | None]:
    checks: list[Check] = []
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    args, model = call_model_loader(module, checkpoint_path, device)

    n_params = sum(int(p.numel()) for p in model.parameters())
    min_params = int(cfg["min_param_count"])
    checks.append(result("Model parameter count", n_params >= min_params,
                         f"{n_params:,} >= {min_params:,}" if n_params >= min_params else f"{n_params:,} < {min_params:,}",
                         n_params=n_params))

    try:
        loss_arange, ce_loss_arange, pos_var, real_vocab = _logits_and_loss_probe(model, args, device, cfg)
    except Exception as exc:
        checks.append(result("forward(x)/forward(x,y) CE agreement", False, f"{type(exc).__name__}: {exc}"))
        return checks, None
    checks.append(result("forward(x)/forward(x,y) CE agreement", True,
                         f"loss_xy={loss_arange:.4f} ce_logits={ce_loss_arange:.4f} pos_var={pos_var:.2e}"))

    checks.append(check_param_cap(model, cfg))   # T1: the PARAM CAP replaces A6's sub-quad gate

    val_tokens, luts, sp = hidden_validation_tokens(args, device)
    seq_len, _ = _probe_shapes(args)
    val_batch_tokens = int(getattr(args, "val_batch_size", DEFAULT_VAL_BATCH_TOKENS) or DEFAULT_VAL_BATCH_TOKENS)
    try:
        val_loss, val_bpb = eval_val(model, val_tokens, luts, device, seq_len, val_batch_tokens)
    except Exception as exc:
        checks.append(result("Held-out evaluation", False, f"eval_val failed: {type(exc).__name__}: {exc}"))
        return checks, None

    floor = float(cfg["quality_floor_val_bpb"])
    min_plausible = float(cfg["min_plausible_val_bpb"])
    checks.append(result("Held-out val_bpb below quality floor", math.isfinite(val_bpb) and val_bpb <= floor,
                         f"val_bpb={val_bpb:.4f} <= floor {floor:.4f}" if val_bpb <= floor else f"val_bpb={val_bpb:.4f} > floor {floor:.4f}",
                         val_bpb=val_bpb, val_loss=val_loss))
    checks.append(result("Held-out val_bpb is plausible (anti-spoof floor)",
                         math.isfinite(val_bpb) and val_bpb >= min_plausible,
                         f"val_bpb={val_bpb:.4f} >= floor {min_plausible:.4f}" if val_bpb >= min_plausible
                         else f"val_bpb={val_bpb:.4f} below floor {min_plausible:.4f} (forged loss or "
                              f"held-out contamination, not a better model)",
                         val_bpb=val_bpb))

    if val_tokens.numel() >= seq_len + 1:
        real_x = val_tokens[:seq_len].reshape(1, -1).to(device=device, dtype=torch.int64).clamp_max(real_vocab - 1)
        real_y = val_tokens[1:seq_len + 1].reshape(1, -1).to(device=device, dtype=torch.int64).clamp_max(real_vocab - 1)
        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    out = model(real_x, real_y)
            if isinstance(out, torch.Tensor) and out.numel() != 1:
                logits = out.float().reshape(-1, out.shape[-1])
                loss_real = float(F.cross_entropy(logits, real_y.reshape(-1), reduction="mean").item())
            else:
                loss_real = float(out.detach().to(torch.float64).item())
        except Exception as exc:
            checks.append(result("OOD: loss(real) < ratio * loss(arange)", False, f"{type(exc).__name__}: {exc}"))
            return checks, None
        ratio = float(cfg["ood_loss_ratio"])
        threshold = ratio * loss_arange
        passed = math.isfinite(loss_real) and loss_real < threshold
        checks.append(result("OOD: loss(real) < ratio * loss(arange)", passed,
                             f"loss(real)={loss_real:.4f} < {ratio}*loss(arange)={threshold:.4f}" if passed
                             else f"loss(real)={loss_real:.4f} >= {ratio}*loss(arange)={threshold:.4f}"))
    # free before the next seed
    del model
    torch.cuda.empty_cache()
    return checks, val_bpb


# -----------------------------------------------------------------------------
# Orchestrate: train N seeds (harness clock) -> eval each -> median bpb
# -----------------------------------------------------------------------------
def run_all(workspace: Path, cfg: dict[str, Any]) -> tuple[dict[str, list[Check]], float | None, list[float] | None]:
    checks: dict[str, list[Check]] = {}
    train_py = workspace / "train_gpt.py"
    if not train_py.exists():
        checks["required_files"] = [result("Required file: train_gpt.py", False, "missing")]
        return checks, None, None
    module = import_solution(train_py)
    if not hasattr(module, "train_for_budget"):
        checks["required_files"] = [result("train_for_budget defined", False,
                                            "train_gpt.py must define train_for_budget(seed, out_dir, budget_seconds)")]
        return checks, None, None
    if not hasattr(module, "load_model_for_verification"):
        checks["required_files"] = [result("load_model_for_verification defined", False,
                                            "train_gpt.py must define load_model_for_verification(path, device)")]
        return checks, None, None
    checks["required_files"] = [result("Entry contract present", True,
                                        "train_for_budget + load_model_for_verification defined")]
    if not torch.cuda.is_available():
        checks["training"] = [result("Training", False, "CUDA is required")]
        return checks, None, None
    if not HOLDOUT_CORPUS_PATH.exists():
        checks["training"] = [result("Held-out corpus available", False,
                                     f"held-out corpus not found at {HOLDOUT_CORPUS_PATH} "
                                     "(the frozen surface must be uploaded fresh at /tests)")]
        return checks, None, None

    seeds = list(cfg.get("seeds") or [1337, 2027, 90101])[: int(cfg["n_seeds"])]
    while len(seeds) < int(cfg["n_seeds"]):
        seeds.append(seeds[-1] + 101)
    per_seed_bpb: list[float] = []
    ckpt_digests: dict[int, str] = {}
    run_root = Path(tempfile.mkdtemp(prefix="e2e_speedrun_"))
    for seed in seeds:
        out_dir = run_root / f"seed_{seed}"
        exists, wall, note = train_one_seed(train_py, seed, out_dir, cfg)
        group: list[Check] = [result(f"[seed {seed}] training produced a checkpoint", exists,
                                      f"wall={wall:.0f}s ({note})" if exists
                                      else f"no model_ckpt.pt after {wall:.0f}s ({note})",
                                      seed=seed, wall_seconds=wall)]
        if exists:
            try:
                ck = out_dir / "model_ckpt.pt"
                ckpt_digests[seed] = hashlib.sha256(ck.read_bytes()).hexdigest()
            except Exception:
                ckpt_digests[seed] = f"unhashable_{seed}"
            try:
                ckpt_checks, bpb = eval_one_checkpoint(module, out_dir / "model_ckpt.pt", cfg)
            except Exception as exc:
                ckpt_checks = [result(f"[seed {seed}] eval", False, f"{type(exc).__name__}: {exc}")]
                bpb = None
            group.extend([result(f"[seed {seed}] {c.name}", c.passed, c.message, hard=c.hard, **c.details)
                          for c in ckpt_checks])
            if bpb is not None and all(c.passed for c in ckpt_checks if c.hard):
                per_seed_bpb.append(bpb)
        checks[f"seed_{seed}"] = group

    # --- cross-seed weight-divergence HARD gate (R2 anti-bake): real seeded
    #     training diverges; byte-identical checkpoints across seeds mean a single
    #     pre-baked checkpoint was returned instead of training within the budget. ---
    digs = list(ckpt_digests.values())
    uniq = len(set(digs))
    div_ok = (len(digs) >= 2 and uniq == len(digs)) or len(digs) <= 1
    checks["cross_seed_divergence"] = [result(
        "Cross-seed weight divergence (anti-baked-checkpoint)", div_ok,
        (f"{uniq}/{len(digs)} seed checkpoints are distinct (seeded training diverges)" if div_ok
         else f"only {uniq}/{len(digs)} distinct checkpoints across seeds -> identical weights returned "
              f"for different seeds (a pre-baked checkpoint, not training within the budget)"),
        hard=True, distinct_checkpoints=uniq, n_checkpoints=len(digs))]

    median_bpb = statistics.median(per_seed_bpb) if len(per_seed_bpb) == len(seeds) else None
    ok = median_bpb is not None and div_ok
    checks["aggregate"] = [result("All seeds produced a real LM within the token+param budget + median computed", ok,
                                  (f"median val_bpb over {len(seeds)} seeds = {median_bpb:.4f} "
                                   f"(per-seed {['%.4f' % b for b in per_seed_bpb]})") if ok
                                  else f"only {len(per_seed_bpb)}/{len(seeds)} seeds produced a passing model",
                                  median_val_bpb=median_bpb, n_seeds=len(seeds), per_seed_bpb=per_seed_bpb)]
    try:
        shutil.rmtree(run_root, ignore_errors=True)
    except Exception:
        pass
    return checks, median_bpb, (per_seed_bpb or None)


# -----------------------------------------------------------------------------
# Reporting + BOUNDED reward (the bench reward spec, performance class)
# -----------------------------------------------------------------------------
def markdown_report(checks: dict[str, list[Check]]) -> str:
    total = sum(len(group) for group in checks.values())
    passed = sum(1 for group in checks.values() for c in group if c.passed)
    lines = ["# Verification Report\n\n", "## Summary\n\n",
             f"- Total checks: {total}\n", f"- Passed: {passed}\n", f"- Failed: {total - passed}\n\n"]
    for category, group in checks.items():
        lines.append(f"## {category.replace('_', ' ').title()}\n\n")
        for c in group:
            status = "PASS" if c.passed else "FAIL"
            details = f" Details: {json.dumps(c.details, sort_keys=True, default=str)}" if c.details else ""
            lines.append(f"- {status}: {c.name} - {c.message}.{details}\n")
        lines.append("\n")
    return "".join(lines)


def _reason_for(check: Check) -> str:
    """Map a failing check onto its reward.md pre-gate reason."""
    name = check.name
    if "Entry contract" in name or "train_for_budget" in name or "load_model_for_verification" in name \
            or "Required file" in name or "training produced a checkpoint" in name \
            or "Held-out corpus" in name or name == "Training":
        return R_BUILD
    if "anti-spoof floor" in name or "anti-baked" in name or "divergence" in name \
            or "plausible" in name:
        return R_CHEAT
    return R_CORRECT


def compute_bounded_reward(checks: dict[str, list[Check]], median_bpb: float | None,
                           cfg: dict[str, Any]) -> tuple[float, list[str], float | None]:
    """reward.md 性能类: reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, range [0, 1].

    Returns (reward, hard_fail_reasons, measured_speedup).

    🔴 `baseline_bpb` and `ref_speedup` are AUTHORING-TIME calibrated CONSTANTS read from the
    frozen manifest. The oracle is NOT in the image and is NEVER run at scoring time. When the
    manifest carries no valid constant the run is a HARD FAIL with an explicit reason — it is
    NEVER silently treated as 1.0."""
    reasons: list[str] = []
    for problem in cfg.get("_manifest_problems") or []:
        if R_BUILD not in reasons:
            reasons.append(R_BUILD)

    # gates 1/2/3 — every failing check contributes its named pre-gate reason
    total = sum(len(group) for group in checks.values())
    for group in checks.values():
        for c in group:
            if not c.passed:
                r = _reason_for(c)
                if r not in reasons:
                    reasons.append(r)
    # a TRUNCATED suite (an early return after a missing entry point / a dead seed) can never
    # score above 0: the full suite must be present as well as green.
    expected = int(cfg.get("expected_total_checks") or 0)
    if expected and total < expected and R_INCOMPLETE not in reasons:
        reasons.append(R_INCOMPLETE)

    # gate 6 — reference solution invalid / uncalibrated. FAIL CLOSED.
    base = cfg.get("baseline_bpb")
    ref = cfg.get("ref_speedup")
    try:
        base = None if base is None else float(base)
    except (TypeError, ValueError):
        base = None
    try:
        ref = None if ref is None else float(ref)
    except (TypeError, ValueError):
        ref = None
    base_ok = base is not None and math.isfinite(base) and base > 0
    ref_ok = ref is not None and math.isfinite(ref) and ref > 1.0
    if not base_ok or not ref_ok or ref < REF_SPEEDUP_AUTHORING_FLOOR:
        if R_BAD_REF not in reasons:
            reasons.append(R_BAD_REF)

    speedup = None
    if base_ok and isinstance(median_bpb, (int, float)) and math.isfinite(median_bpb) and median_bpb > 0:
        speedup = float(base) / float(median_bpb)

    # gate 5 — did not beat the baseline recipe it started from
    if speedup is None or speedup <= 1.0:
        if R_NO_GAIN not in reasons:
            reasons.append(R_NO_GAIN)

    if reasons:
        return 0.0, reasons, speedup
    reward = min(1.0, max(0.0, max(0.0, min(1.0, math.log(speedup) / math.log(float(ref)) - 1.0))))
    return reward, [], speedup


def emit(reward: float, reasons: list[str], cfg: dict[str, Any],
         checks: dict[str, list[Check]] | None = None,
         median_bpb: float | None = None, speedup: float | None = None,
         per_seed_bpb: list[float] | None = None, extra: dict[str, Any] | None = None,
         workspace: Path | None = None) -> None:
    """SINGLE writer for the whole output contract, so hard-fail and success paths cannot
    diverge in schema or in file count."""
    checks = checks or {}
    reward = float(reward)
    if not math.isfinite(reward):
        reward, reasons = 0.0, list(reasons) + [R_UNNAMED]
    reward = min(1.0, max(0.0, reward))
    reasons = [r for i, r in enumerate(reasons) if r not in reasons[:i]]
    if reward == 0.0 and not reasons:
        reasons = [R_UNNAMED]
    if reasons:
        reward = 0.0

    total = sum(len(group) for group in checks.values())
    passed = sum(1 for group in checks.values() for c in group if c.passed)
    expected = int(cfg.get("expected_total_checks") or 0)
    cand_cv = None
    if per_seed_bpb and len(per_seed_bpb) >= 2:
        mean = statistics.fmean(per_seed_bpb)
        if mean > 0:
            cand_cv = statistics.pstdev(per_seed_bpb) / mean

    ref = cfg.get("ref_speedup")
    base = cfg.get("baseline_bpb")
    oracle_bpb = None
    try:
        if base is not None and ref not in (None, 0):
            oracle_bpb = float(base) / float(ref)
    except (TypeError, ValueError, ZeroDivisionError):
        oracle_bpb = None

    desc = {"metric_kind": "quality_at_fixed_budget", "metric_name": "val_bpb",
            "metric_direction": "lower_is_better", "timing_measured": False}
    reward_json = {
        "task_type": "performance",
        "reward": reward,
        "hard_fail_reasons": reasons,
        "speedup": speedup,
        "ref_speedup": ref,
        "cv": {"baseline": cfg.get("baseline_cv"), "candidate": cand_cv},
        **desc,
    }
    correctness = {
        "passed": passed, "total": total, "expected_total": expected,
        "all_passed": bool(expected and total == expected and passed == total),
        "hard_fail_reasons": reasons,
        "failed_checks": [{"category": cat, "name": c.name, "message": c.message,
                           "gate": _reason_for(c), "details": c.details}
                          for cat, group in checks.items() for c in group if not c.passed],
    }
    benchmark = {
        **desc,
        "candidate_median_val_bpb": median_bpb,
        "candidate_val_bpb": median_bpb,
        "per_seed_val_bpb": per_seed_bpb,
        "n_seeds": cfg.get("n_seeds"),
        "token_budget": cfg.get("token_budget"),
        "wallclock_guard_seconds": cfg.get("budget_seconds"),
        "max_param_count": cfg.get("max_param_count"),
        "baseline_bpb_frozen": base,
        "oracle_bpb_frozen": oracle_bpb,
        "ref_speedup_frozen": ref,
        "speedup": speedup,
        "oracle_executed_by_scorer": False,
        "baseline_executed_by_scorer": False,
        "metric_void_on_hard_fail": bool(reasons),
    }
    metrics = {
        "task_type": "performance",
        "reward": reward,
        "partial_score": reward,
        "hard_fail_reasons": reasons,
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "reward_spec": "the bench reward spec (performance class, bounded log form)",
        "reward_form": (
            "reward.md 性能类: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 in [0,1]; "
            "speedup = baseline_bpb / median_val_bpb at a FIXED TRAINING-TOKEN budget; "
            "ref_speedup is a FROZEN authoring-time constant (baseline_bpb / oracle_val_bpb) and "
            "the oracle is NEVER run at scoring. 0 at or below the tuned-AdamW recipe the solver "
            "started from, 0.5 at the demonstrated in-budget ceiling, 1.0 only at "
            "speedup = ref_speedup**2."),
        "speedup_semantics": (
            "the `speedup` field carries baseline_bpb / median_val_bpb at a FIXED TRAINING-TOKEN "
            "budget -- a QUALITY ratio (bits-per-byte, lower is better), not a wall-clock ratio. "
            "Nothing in this task is timed for score; the wall-clock is only a feasibility guard. "
            "reward.md's bounded log form is ratio-agnostic; only the variable's name says latency."),
        "speedup": speedup,
        "ref_speedup": ref,
        "baseline_bpb": base,
        "median_val_bpb": median_bpb,
        "per_seed_bpb": per_seed_bpb,
        "budget_axis": "training_tokens (harness-enforced BY CONSTRUCTION; never self-reported)",
        "token_budget": cfg.get("token_budget"),
        "wallclock_guard_seconds": cfg.get("budget_seconds"),
        "max_param_count": cfg.get("max_param_count"),
        "n_seeds": cfg.get("n_seeds"),
        "passed": passed, "total": total, "tests": {"passed": passed, "total": total},
        "tests_expected_total": expected,
        "pass_rate": (passed / total if total else 0.0),
        "cv": reward_json["cv"],
        **desc,
        "failed_checks": correctness["failed_checks"],
    }
    verifier_state = {
        "task_id": "e2e-a4-token-efficiency-budget",
        "task_type": "performance",
        "reward": reward,
        "hard_fail_reasons": reasons,
        "speedup": speedup, "ref_speedup": ref, "baseline_bpb": base,
        "median_val_bpb": median_bpb,
        "passed": passed, "total": total, "expected_total": expected,
        "ts": time.time(),
    }
    if extra:
        metrics.update(extra)
        verifier_state["detail"] = extra

    if workspace is not None:
        try:
            (workspace / "verification_report.md").write_text(markdown_report(checks), encoding="utf-8")
        except Exception:
            pass
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
    print(f"REWARD={reward:.6f} speedup={speedup} ref_speedup={ref} baseline_bpb={base} "
          f"median_val_bpb={median_bpb} hard_fail_reasons={reasons}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-zero", metavar="REASON", default=None)
    ap.add_argument("--detail", default="")
    ns, _ = ap.parse_known_args()
    cfg = load_manifest()

    if ns.emit_zero:
        known = {R_BUILD, R_CORRECT, R_CHEAT, R_FORBIDDEN, R_NO_GAIN, R_BAD_REF,
                 R_CRASH, R_UNNAMED, R_INCOMPLETE}
        reason = ns.emit_zero.strip()
        if reason not in known:
            reason = R_BUILD
        detail = ns.detail or ns.emit_zero
        emit(0.0, [reason], cfg,
             {"harness": [result("Harness pre-flight", False, detail)]},
             extra={"harness_message": detail})
        return 1

    extra: dict[str, Any] = {}
    pre_reasons: list[str] = []
    if cfg.get("_manifest_problems"):
        extra["manifest_problems"] = cfg["_manifest_problems"]
    # Refuse to score against an in-image (solver-reachable) evaluation surface: after the
    # de-leak rebuild there is no /opt/verifier at all, so its presence means a stale image.
    if Path("/opt/verifier").exists():
        pre_reasons.append(R_FORBIDDEN)
        extra["stale_baked_verifier"] = (
            "/opt/verifier exists in this image — the frozen evaluation surface must be uploaded "
            "fresh at /tests only; refusing to score against a solver-reachable copy")
    for baked in ("/opt/strong_baseline", "/opt/naive", "/opt/negative", "/opt/ceiling"):
        if Path(baked).exists():
            pre_reasons.append(R_FORBIDDEN)
            extra.setdefault("baked_reference_recipes", []).append(baked)

    workspace = WORKSPACE if WORKSPACE.exists() else Path.cwd()
    sanitize_python_path(workspace)
    try:
        checks, median_bpb, per_seed = run_all(workspace, cfg)
    except Exception as exc:
        checks = {"verifier_error": [result("Verifier error", False, f"{type(exc).__name__}: {exc}")]}
        median_bpb, per_seed = None, None
    if _TAMPER["n_files"]:
        extra["frozen_surface"] = dict(_TAMPER)
    if _PRIV_OBSERVED:
        extra["training_privileges"] = list(_PRIV_OBSERVED)
        undropped = [r for r in _PRIV_OBSERVED if r.get("requested_uid") and not r.get("dropped")]
        if undropped:
            # not a candidate cheat, but the quarantines were weaker than designed for those
            # seeds: surface it loudly instead of scoring as if they had held.
            extra["training_privileges_WARNING"] = (
                f"{len(undropped)}/{len(_PRIV_OBSERVED)} seeds did NOT run under the requested "
                "unprivileged uid — the corpus/held-out quarantine was nominal for them")
    reward, reasons, speedup = compute_bounded_reward(checks, median_bpb, cfg)
    reasons = pre_reasons + reasons
    if _TAMPER["changed"]:
        for r in (R_CHEAT, R_FORBIDDEN):
            if r not in reasons:
                reasons.append(r)
    if _TAMPER["mode_changed"] or _TAMPER["unexpected_files"]:
        if R_CHEAT not in reasons:
            reasons.append(R_CHEAT)
    if reasons:
        reward = 0.0
    emit(reward, reasons, cfg, checks, median_bpb, speedup, per_seed, extra, workspace)
    print(markdown_report(checks))
    return 0 if (reward > 0.0 and not reasons) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:            # never exit without the output contract
        try:
            emit(0.0, [R_CRASH], dict(_DEFAULTS),
                 {"verifier": [result("Verifier crashed", False, f"{type(exc).__name__}: {exc}")]})
        finally:
            raise SystemExit(1)
