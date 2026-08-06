#!/usr/bin/env python3
"""/opt/loop/private/dev_eval.py — the per-round DEV proxy for e2e-a3-moe-train-budget.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh (0700). The solver's uid cannot read it.
Adapted from the a4-token-efficiency-budget dev_eval (its sibling); the ONLY substantive difference
is the entry contract: a3's training is wall-clock-budgeted and launched via run_training.sh, whereas
a4's is token-budgeted via train_for_budget().

WHAT IT DOES (a faithful, CHEAP mirror of the graded path, on PUBLIC data):
  1. Slices a SMALL public token shard out of the PUBLIC corpus the solver already has
     (/data/datasets/wikitext_bpe8k/train.bin) into an isolated dev DATA_PATH.
  2. Runs the candidate's run_training.sh from an ISOLATED staging dir (source only; weight/data
     blobs stripped) UNDER THE uid-65534 PRIVILEGE DROP the graded path uses, with a SHORT dev
     wall-clock budget (dev_wallclock_sec) and a fixed dev seed.
  3. Reloads the checkpoint via the candidate's load_model_for_verification and runs a
     FIRST-PRINCIPLES correctness self-check (total-param floor in the SAME [floor] the grade
     enforces, forward(x)/forward(x,y) CE agreement, logits vary across positions, a plausible
     val_bpb band) — no reference recipe and no anchor are needed to decide pass/fail.
  4. Computes val_bpb on the PUBLIC val split (/data/datasets/wikitext_bpe8k/val.bin) using the same
     SentencePiece byte-LUT machinery as the grade, so the number is comparable.

WHAT IT NEVER TOUCHES: /tests, the held-out corpus, the calibrated baseline_bpb / ref_speedup, or ANY
environment-supplied threshold. The dev signal is an ABSOLUTE val_bpb, never normalised against the
graded anchor. Constants come from /opt/loop/private/manifest.json ONLY.

🔴 GPU/REDUCED-WORKLOAD + FIDELITY NOTE: unlike a4 (whose budget is TOKENS and is therefore
node-speed-immune), a3's budget is WALL-CLOCK. The dev proxy runs a SHORT dev wall-clock
(dev_wallclock_sec, far below the graded budget) so a per-round pass is minutes not the full graded
run — but that makes the dev val_bpb a function of BOTH the recipe AND this node's speed. So the dev
val_bpb is only a COARSE direction signal: a recipe that trains more efficiently should reach a lower
val_bpb in the short dev budget, but the absolute number is not comparable across nodes and is not the
graded score. best-of-k still prefers the lower-dev-val_bpb correctness-passing snapshot; the
authoritative score is the runner end-of-session grade at the full budget with the calibrated anchor.

OUTPUT: /logs/loop/dev/{verifier_state.json, reward.json}; on infra failure, harness_error.txt.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

LOOP_PRIVATE = Path("/opt/loop/private")
MANIFEST = LOOP_PRIVATE / "manifest.json"
DEV_OUT = Path("/logs/loop/dev")
DEV_OUT.mkdir(parents=True, exist_ok=True)

INV_SUBMISSION_MISSING = "submission_missing"
INV_NO_CHECKPOINT = "no_checkpoint_produced"
INV_PARAM_FLOOR = "param_below_floor"
INV_CE_AGREE = "forward_ce_disagreement"
INV_LOGITS_CONST = "logits_constant_across_positions"
INV_VAL_IMPLAUSIBLE = "val_bpb_implausible"
INV_VAL_FLOOR = "val_bpb_above_quality_floor"
INV_LOAD = "load_model_for_verification_failed"
INV_HARNESS = "harness_error"

_WEIGHT_EXT = {".pt", ".pth", ".ptz", ".bin", ".safetensors", ".ckpt", ".npz",
               ".npy", ".pkl", ".pickle", ".h5", ".onnx", ".gguf", ".pt2"}


def _cfg() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write_state(correctness_ok, failing_invariant, dev_val_bpb, dev_reward,
                 uid_dropped, observed_euid, extra=None):
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {"correctness_ok": bool(correctness_ok), "hard_fail_reasons": reasons,
             "failing_invariant": failing_invariant or "",
             "dev_uid_dropped": uid_dropped, "observed_euid": observed_euid}
    if extra:
        state.update(extra)
    reward = {"dev_score": float(dev_reward),
              "dev_metric": "val_bpb (PUBLIC dev proxy; lower is better; NOT your graded score)",
              "dev_val_bpb": dev_val_bpb, "correctness_ok": bool(correctness_ok)}
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, None, 0.0, None, None, {"harness_message": msg})


def build_dev_shard(corpus_dir: Path, shard_dir: Path, n_tokens: int, offset: int) -> bool:
    import numpy as np
    shard_dir.mkdir(parents=True, exist_ok=True)
    src = corpus_dir / "train.bin"
    if not src.exists():
        return False
    d = np.memmap(src, dtype=np.uint16, mode="r")
    have = int(d.shape[0])
    if have < n_tokens:
        return False
    off = offset if (0 <= offset <= have - n_tokens) else 0
    np.asarray(d[off:off + n_tokens], dtype=np.uint16).tofile(shard_dir / "train.bin")
    for extra in ("val.bin", "meta.pkl"):
        sp = corpus_dir / extra
        if sp.exists():
            shutil.copy2(sp, shard_dir / extra)
    return (shard_dir / "train.bin").stat().st_size // 2 == n_tokens


def train_candidate(cfg: dict, submission_dir: Path, shard_dir: Path, out_dir: Path):
    """Run the candidate's run_training.sh on the dev shard under the uid drop and a SHORT dev
    wall-clock budget. Returns (checkpoint_or_None, uid_dropped, observed_euid, note, stderr_tail)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / "_stage_src"
    stage.mkdir(parents=True, exist_ok=True)
    src_root = submission_dir
    try:
        for p in src_root.rglob("*"):
            if not p.is_file() or "_stage_src" in p.parts:
                continue
            if p.suffix.lower() in _WEIGHT_EXT or p.stat().st_size > 8 * 1024 * 1024:
                continue
            rel = p.relative_to(src_root)
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, stage / rel)
    except Exception:
        pass
    if not (stage / "run_training.sh").exists():
        return None, False, None, "run_training.sh missing", ""

    guard = float(cfg["dev_wallclock_sec"])
    grace = float(cfg.get("dev_guard_grace_seconds", 30.0))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["SEED"] = str(int(cfg["dev_seed"]))
    env["PARAM_FLOOR"] = str(int(cfg["dev_param_floor"]))
    env["WALLCLOCK_SEC"] = str(guard)
    env["OUT_CKPT"] = str(out_dir / "model_ckpt.pt")
    env["DATA_PATH"] = str(shard_dir)
    env["TOKENIZER_PATH"] = str(cfg["tokenizer_path"])
    env["NANOGPT_REPO"] = str(cfg["nanogpt_repo"])
    env["PATH"] = "/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/bin:/bin"

    train_uid = cfg.get("dev_uid")
    def preexec_plain():
        os.setsid()
    preexec = preexec_plain
    if train_uid:
        try:
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
            os.chmod(shard_dir, 0o755)
            for p in shard_dir.iterdir():
                try:
                    os.chmod(p, 0o644)
                except Exception:
                    pass
        except Exception:
            pass
        home = out_dir / "_devhome"
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
        env["DEV_UID_REPORT"] = str(out_dir / "_dev_uid.txt")
        _uid = int(train_uid)
        def preexec_drop():
            os.setsid()
            try:
                os.setgroups([])
                os.setgid(_uid)
                os.setuid(_uid)
            except Exception:
                pass
        preexec = preexec_drop

    # record euid via a tiny wrapper the child writes before exec'ing run_training.sh
    proc = subprocess.Popen(
        ["bash", "-c",
         'if [ -n "${DEV_UID_REPORT:-}" ]; then id -u >/dev/null; python3 -c "import os;open(os.environ[\'DEV_UID_REPORT\'],\'w\').write(\'%d %d\'%(os.getuid(),os.geteuid()))" 2>/dev/null || true; fi; '
         'exec bash "$0"', str(stage / "run_training.sh")],
        cwd=str(stage), env=env, preexec_fn=preexec,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    note = "completed"
    stderr_tail = b""
    try:
        _, stderr_tail = proc.communicate(timeout=guard + grace)
    except subprocess.TimeoutExpired:
        note = "hard-killed at dev guard+grace"
        for sg in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sg)
            except ProcessLookupError:
                pass
            time.sleep(3)
        try:
            _, stderr_tail = proc.communicate(timeout=20)
        except Exception:
            pass
    observed_euid = None
    try:
        observed_euid = (out_dir / "_dev_uid.txt").read_text().split()[1]
    except Exception:
        observed_euid = None
    uid_dropped = bool(train_uid) and observed_euid == str(int(train_uid)) if train_uid else False
    ckpt = out_dir / "model_ckpt.pt"
    try:
        if ckpt.exists():
            os.chmod(ckpt, 0o644)
    except Exception:
        pass
    stderr_txt = (stderr_tail or b"").decode("utf-8", "ignore")
    return (ckpt if ckpt.exists() else None), uid_dropped, observed_euid, note, stderr_txt


def evaluate(cfg: dict, submission_dir: Path, ckpt: Path):
    import importlib.util
    import numpy as np
    import sentencepiece as spm
    import torch
    import torch.nn.functional as F
    import inspect

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    train_gpt = submission_dir / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("dev_reload_train_gpt", train_gpt)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dev_reload_train_gpt"] = module
    sub_parent = str(submission_dir.resolve())
    if sub_parent not in sys.path:
        sys.path.insert(0, sub_parent)
    if str(cfg["nanogpt_repo"]) not in sys.path:
        sys.path.insert(0, str(cfg["nanogpt_repo"]))
    spec.loader.exec_module(module)

    loader = getattr(module, "load_model_for_verification", None)
    if loader is None:
        return False, INV_LOAD, None, "load_model_for_verification missing"
    args = module.Hyperparameters() if hasattr(module, "Hyperparameters") else None
    try:
        sig = inspect.signature(loader)
        kwargs, positional = {}, []
        for name, par in sig.parameters.items():
            if name in {"checkpoint_path", "path", "model_path"}:
                kwargs[name] = ckpt
            elif name == "device":
                kwargs[name] = device
            elif par.default is inspect._empty:
                if not positional:
                    positional.append(ckpt)
                elif len(positional) == 1:
                    positional.append(device)
        loaded = loader(*positional, **kwargs)
    except Exception as exc:
        return False, INV_LOAD, None, f"{type(exc).__name__}: {exc}"
    if isinstance(loaded, tuple) and len(loaded) == 2:
        loaded_args, model = loaded
        if loaded_args is not None:
            args = loaded_args
    else:
        model = loaded
    if not isinstance(model, torch.nn.Module):
        return False, INV_LOAD, None, "loader did not return an nn.Module / (args, model)"
    model = model.to(device).eval()

    # total-parameter FLOOR — the SAME capacity floor the grade re-counts (weight-tie deduplicated).
    seen, unique_params = set(), 0
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        unique_params += int(p.numel())
    floor = int(cfg["dev_param_floor"])
    if unique_params < floor:
        return False, INV_PARAM_FLOOR, None, f"{unique_params:,} < floor {floor:,}"

    seq_len = int(getattr(args, "train_seq_len", 512) or 512)
    seq_len = max(seq_len, 16)
    vocab_size = int(getattr(args, "vocab_size", 8192) or 8192)

    arange = torch.arange(seq_len, device=device, dtype=torch.int64)
    x_ar = (arange % vocab_size).reshape(1, -1)
    y_ar = ((arange + 1) % vocab_size).reshape(1, -1)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x_ar)
            loss_xy = model(x_ar, y_ar)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.dtype.is_floating_point:
        return False, INV_CE_AGREE, None, "forward(x) did not return float logits (B,T,V)"
    lf = logits.float()
    pos_var = float(lf.var(dim=1).mean().item())
    if not math.isfinite(pos_var) or pos_var <= float(cfg["dev_logits_position_variance_floor"]):
        return False, INV_LOGITS_CONST, None, f"logits constant across positions (var={pos_var:.2e})"
    real_vocab = logits.shape[-1]
    ce = float(F.cross_entropy(lf.reshape(-1, real_vocab),
                               y_ar.reshape(-1).clamp_max(real_vocab - 1), reduction="mean").item())
    lxy = float(loss_xy.detach().to(torch.float64).item()) if isinstance(loss_xy, torch.Tensor) else float(loss_xy)
    if not math.isfinite(lxy) or abs(lxy - ce) > float(cfg["dev_logits_loss_agreement_tol"]):
        return False, INV_CE_AGREE, None, f"forward(x,y)={lxy:.4f} != CE(forward(x),y)={ce:.4f}"

    tok_path = Path(str(getattr(args, "tokenizer_path", cfg["tokenizer_path"])))
    if not tok_path.exists():
        tok_path = Path(cfg["tokenizer_path"])
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    sp_v = int(sp.vocab_size())
    table = max(sp_v, vocab_size)
    base_bytes = np.zeros((table,), dtype=np.int32)
    has_lead = np.zeros((table,), dtype=np.bool_)
    is_bound = np.ones((table,), dtype=np.bool_)
    for tid in range(sp_v):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_bound[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_lead[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    base_bytes_lut = torch.tensor(base_bytes, dtype=torch.int32, device=device)
    has_lead_lut = torch.tensor(has_lead, dtype=torch.bool, device=device)
    is_bound_lut = torch.tensor(is_bound, dtype=torch.bool, device=device)

    val_path = Path(cfg["corpus_dir"]) / "val.bin"
    if not val_path.exists():
        return False, INV_VAL_IMPLAUSIBLE, None, f"public val split missing at {val_path}"
    vd = np.memmap(val_path, dtype=np.uint16, mode="r")
    cap_tok = int(cfg["dev_val_tokens"])
    ids = np.asarray(vd[: min(len(vd), cap_tok + 1)], dtype=np.int64)
    usable = ((ids.size - 1) // seq_len) * seq_len
    if usable <= 0:
        return False, INV_VAL_IMPLAUSIBLE, None, "public val split too short for seq_len"
    val_tokens = torch.from_numpy(ids[: usable + 1]).contiguous()
    batch_seqs = max(32768 // seq_len, 1)
    total_seqs = usable // seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for s0 in range(0, total_seqs, batch_seqs):
            s1 = min(s0 + batch_seqs, total_seqs)
            r0, r1 = s0 * seq_len, s1 * seq_len + 1
            local = val_tokens[r0:r1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = model(x, y)
            if isinstance(out, torch.Tensor) and out.numel() != 1:
                lg = out.float().reshape(-1, out.shape[-1])
                bl = F.cross_entropy(lg, y.reshape(-1).clamp_max(lg.shape[-1] - 1), reduction="mean")
            else:
                bl = out
            ntok = float(y.numel())
            loss_sum += bl.detach().to(torch.float64) * ntok
            tok_count += ntok
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int32)
            tb = tb + (has_lead_lut[tgt] & ~is_bound_lut[prev]).to(torch.int32)
            byte_count += tb.to(torch.float64).sum()
    val_loss = (loss_sum / tok_count).item()
    bpb = float((val_loss / math.log(2.0)) * (tok_count.item() / byte_count.item()))
    if not math.isfinite(bpb) or bpb < float(cfg["dev_min_plausible_val_bpb"]):
        return False, INV_VAL_IMPLAUSIBLE, bpb if math.isfinite(bpb) else None, \
            f"val_bpb={bpb} below plausibility floor (forged loss / contamination)"
    if bpb > float(cfg["dev_quality_floor_val_bpb"]):
        return False, INV_VAL_FLOOR, bpb, f"val_bpb={bpb:.4f} above dev quality floor"
    return True, None, bpb, f"ok params={unique_params:,} val_bpb={bpb:.4f}"


def main() -> int:
    try:
        cfg = _cfg()
    except Exception as exc:
        _harness_error(f"could not read dev manifest: {type(exc).__name__}: {exc}")
        return 0
    submission_dir = Path(cfg["submission_dir"])
    if not (submission_dir / "run_training.sh").exists() or not (submission_dir / "train_gpt.py").exists():
        _write_state(False, INV_SUBMISSION_MISSING, None, 0.0, None, None,
                     {"detail": f"run_training.sh / train_gpt.py missing under {submission_dir}"})
        return 0
    import tempfile
    run_root = Path(tempfile.mkdtemp(prefix="a3_devloop_"))
    try:
        shard_dir = run_root / "_dev_shard"
        if not build_dev_shard(Path(cfg["corpus_dir"]), shard_dir,
                               int(cfg["dev_train_tokens"]), int(cfg["dev_train_offset"])):
            _harness_error("could not build the public dev shard (corpus missing/too small)")
            return 0
        out_dir = run_root / "dev_seed"
        ckpt, uid_dropped, observed_euid, note, stderr_txt = train_candidate(
            cfg, submission_dir, shard_dir, out_dir)
        if ckpt is None:
            infra = any(s in (stderr_txt or "") for s in
                        ("ModuleNotFoundError", "ImportError", "Traceback (most recent call last)"))
            if infra and "run_training.sh missing" not in (stderr_txt or ""):
                _harness_error("dev training subprocess raised an infra error before producing a "
                               f"checkpoint: {(stderr_txt or '').strip().splitlines()[-1:]}")
                return 0
            _write_state(False, INV_NO_CHECKPOINT, None, 0.0, uid_dropped, observed_euid, {"note": note})
            return 0
        try:
            ok, inv, bpb, detail = evaluate(cfg, submission_dir, ckpt)
        except Exception as exc:
            _harness_error(f"dev evaluate() raised: {type(exc).__name__}: {exc}")
            return 0
        # dev_score: HIGHER is better for best-of-k; val_bpb is lower-is-better, so use 1/bpb.
        dev_reward = (1.0 / bpb) if (ok and bpb and bpb > 0) else 0.0
        _write_state(ok, inv, bpb, dev_reward, uid_dropped, observed_euid, {"detail": detail})
        return 0
    finally:
        try:
            shutil.rmtree(run_root, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
