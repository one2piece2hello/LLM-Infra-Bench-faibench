#!/usr/bin/env python3
"""/opt/loop/private/dev_eval.py — the per-round DEV proxy for e2e-a4-token-efficiency-budget.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh (which is 0700). The solver's
dropped uid cannot read this file.

WHAT IT DOES (a faithful, CHEAP, node-speed-immune mirror of the graded path, on PUBLIC data):
  1. Slices a SMALL public token shard (dev_train_tokens, deterministic window) out of the
     PUBLIC corpus the solver already has (/data/datasets/wikitext_bpe8k/train.bin) and points
     DATA_PATH at it. The budget is TOKENS, not wall-clock, so the val_bpb it yields is a stable
     function of the recipe+data+seed and does not drift with node speed.
  2. Runs the candidate's train_for_budget(dev_seed, out_dir, dev_guard) from an ISOLATED
     staging dir (source only; weight/data blobs stripped, so a smuggled checkpoint cannot game
     even the dev signal), UNDER THE uid-65534 PRIVILEGE DROP the graded path uses.
  3. Reloads the checkpoint via the candidate's load_model_for_verification and runs a
     FIRST-PRINCIPLES correctness self-check (param count in the SAME [floor,cap] the grade
     enforces, forward(x)/forward(x,y) CE agreement, logits vary across positions, a plausible
     val_bpb band) — no reference implementation and no anchor are needed to decide pass/fail.
  4. Computes val_bpb on the PUBLIC val split (/data/datasets/wikitext_bpe8k/val.bin — the
     "local progress monitor" the task statement already gives the solver), using the same
     SentencePiece byte-LUT machinery as the grade so the number is comparable.

WHAT IT NEVER TOUCHES: /tests, the held-out corpus, the calibrated baseline_bpb / ref_speedup,
or ANY environment-supplied threshold. The dev signal is an ABSOLUTE val_bpb, never normalised
against the graded anchor. Constants come from /opt/loop/private/manifest.json ONLY.

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

# ---- named DEV failing-invariants (mirror the graded gate vocabulary, dev-scoped) ----
INV_SUBMISSION_MISSING = "submission_missing"
INV_NO_CHECKPOINT = "no_checkpoint_produced"
INV_PARAM_FLOOR = "param_below_floor"
INV_PARAM_CAP = "param_above_cap"
INV_CE_AGREE = "forward_ce_disagreement"
INV_LOGITS_CONST = "logits_constant_across_positions"
INV_VAL_IMPLAUSIBLE = "val_bpb_implausible"
INV_VAL_FLOOR = "val_bpb_above_quality_floor"
INV_LOAD = "load_model_for_verification_failed"
INV_HARNESS = "harness_error"


def _cfg() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write_state(correctness_ok: bool, failing_invariant: str | None,
                 dev_val_bpb: float | None, dev_reward: float,
                 uid_dropped: bool | None, observed_euid, extra: dict | None = None) -> None:
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {
        "correctness_ok": bool(correctness_ok),
        "hard_fail_reasons": reasons,
        "failing_invariant": failing_invariant or "",
        "dev_uid_dropped": uid_dropped,
        "observed_euid": observed_euid,
    }
    if extra:
        state.update(extra)
    reward = {
        # dev_score: higher is better, used ONLY to rank best-of-k. dev_val_bpb is the
        # human-facing absolute proxy (lower is better). NEITHER is the graded reward.
        "dev_score": float(dev_reward),
        "dev_metric": "val_bpb (PUBLIC dev proxy; lower is better; NOT your graded score)",
        "dev_val_bpb": dev_val_bpb,
        "correctness_ok": bool(correctness_ok),
    }
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, None, 0.0, None, None, {"harness_message": msg})


# -----------------------------------------------------------------------------
# dev shard from the PUBLIC corpus (deterministic window; public data, no quarantine needed)
# -----------------------------------------------------------------------------
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


_DEV_RUNNER = r'''
import importlib.util, os, sys
sub = sys.argv[1]; seed = int(sys.argv[2]); out_dir = sys.argv[3]; budget = float(sys.argv[4])
rep = os.environ.get("DEV_UID_REPORT")
if rep:
    try: open(rep, "w").write("%d %d\n" % (os.getuid(), os.geteuid()))
    except Exception: pass
spec = importlib.util.spec_from_file_location("dev_train_gpt", sub)
mod = importlib.util.module_from_spec(spec); sys.modules["dev_train_gpt"] = mod
sys.path.insert(0, os.path.dirname(os.path.abspath(sub)))
spec.loader.exec_module(mod)
fn = getattr(mod, "train_for_budget", None)
if fn is None:
    print("train_for_budget missing", file=sys.stderr); sys.exit(7)
fn(seed=seed, out_dir=out_dir, budget_seconds=budget)
'''

_WEIGHT_EXT = {".pt", ".pth", ".ptz", ".bin", ".safetensors", ".ckpt", ".npz",
               ".npy", ".pkl", ".pickle", ".h5", ".onnx", ".gguf", ".pt2"}


def train_candidate(cfg: dict, submission_py: Path, shard_dir: Path, out_dir: Path):
    """Train the candidate recipe on the dev shard under the uid drop. Returns
    (checkpoint_path_or_None, uid_dropped, observed_euid, note)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / "_stage_src"
    stage.mkdir(parents=True, exist_ok=True)
    src_root = submission_py.parent
    try:
        for p in src_root.rglob("*"):
            if not p.is_file() or "_stage_src" in p.parts or p.name.startswith("_dev_runner"):
                continue
            if p.suffix.lower() in _WEIGHT_EXT or p.stat().st_size > 8 * 1024 * 1024:
                continue
            rel = p.relative_to(src_root)
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, stage / rel)
    except Exception:
        pass
    staged_py = stage / submission_py.name
    if not staged_py.exists():
        shutil.copy2(submission_py, staged_py)
    (stage / "_dev_runner.py").write_text(_DEV_RUNNER, encoding="utf-8")

    guard = float(cfg["dev_guard_seconds"])
    grace = float(cfg["dev_guard_grace_seconds"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["SEED"] = str(int(cfg["dev_seed"]))
    env["TOKEN_BUDGET"] = str(int(cfg["dev_train_tokens"]))
    env["MAX_WALLCLOCK_SEC"] = str(guard)
    env["MAX_PARAMS"] = str(int(cfg["dev_param_cap"]))
    env["OUT_CKPT"] = str(out_dir / "model_ckpt.pt")
    env["DATA_PATH"] = str(shard_dir)
    env["TOKENIZER_PATH"] = str(cfg["tokenizer_path"])
    env["NANOGPT_REPO"] = str(cfg["nanogpt_repo"])

    py = sys.executable or shutil.which("python3") or "python3"
    train_uid = cfg.get("dev_uid")

    def preexec_plain():
        os.setsid()

    preexec = preexec_plain
    if train_uid:
        # the WHOLE ancestor chain must be traversable by the unprivileged uid (mkdtemp makes a
        # 0700 root parent), then grant it its staging + shard + out_dir and nothing else.
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

    t0 = time.monotonic()
    proc = subprocess.Popen(
        [py, str(stage / "_dev_runner.py"), str(staged_py), str(cfg["dev_seed"]),
         str(out_dir), str(guard)],
        cwd=str(stage), env=env, preexec_fn=preexec,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
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
    wall = time.monotonic() - t0

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
    return (ckpt if ckpt.exists() else None), uid_dropped, observed_euid, f"{note} ({wall:.0f}s)", stderr_txt


# -----------------------------------------------------------------------------
# reload + first-principles self-check + PUBLIC val_bpb (byte-LUT, lifted compact)
# -----------------------------------------------------------------------------
def evaluate(cfg: dict, submission_py: Path, ckpt: Path):
    import numpy as np
    import sentencepiece as spm
    import torch
    import torch.nn.functional as F
    import importlib.util

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    spec = importlib.util.spec_from_file_location("dev_reload_train_gpt", submission_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dev_reload_train_gpt"] = module
    sub_parent = str(submission_py.parent.resolve())
    if sub_parent not in sys.path:
        sys.path.insert(0, sub_parent)
    if str(cfg["nanogpt_repo"]) not in sys.path:
        sys.path.insert(0, str(cfg["nanogpt_repo"]))
    spec.loader.exec_module(module)

    loader = getattr(module, "load_model_for_verification", None)
    if loader is None:
        return False, INV_LOAD, None, "load_model_for_verification missing"
    import inspect
    args = module.Hyperparameters() if hasattr(module, "Hyperparameters") else None
    try:
        sig = inspect.signature(loader)
        kwargs, positional = {}, []
        for name, par in sig.parameters.items():
            if name in {"checkpoint_path", "path", "model_path", "compressed_checkpoint_path"}:
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

    # param cap/floor — the SAME band the grade enforces (does not coach a sub-floor model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    floor = int(cfg["dev_param_floor"])
    cap = int(cfg["dev_param_cap"])
    if n_params < floor:
        return False, INV_PARAM_FLOOR, None, f"{n_params:,} < floor {floor:,}"
    if n_params > cap:
        return False, INV_PARAM_CAP, None, f"{n_params:,} > cap {cap:,}"

    seq_len = int(getattr(args, "train_seq_len", 1024) or 1024)
    seq_len = max(seq_len, 16)
    vocab_size = int(getattr(args, "vocab_size", 8192) or 8192)

    # forward(x)/forward(x,y) CE agreement + logits position variance
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

    # ---- val_bpb on the PUBLIC val split (byte-LUT SentencePiece, same as the grade) ----
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
    n = val_tokens.numel()
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
    return True, None, bpb, f"ok params={n_params:,} val_bpb={bpb:.4f}"


def main() -> int:
    try:
        cfg = _cfg()
    except Exception as exc:
        _harness_error(f"could not read dev manifest: {type(exc).__name__}: {exc}")
        return 0

    submission_py = Path(cfg["submission_dir"]) / "train_gpt.py"
    if not submission_py.exists():
        _write_state(False, INV_SUBMISSION_MISSING, None, 0.0, None, None,
                     {"detail": f"{submission_py} not found"})
        return 0

    import tempfile
    run_root = Path(tempfile.mkdtemp(prefix="a4_devloop_"))
    try:
        shard_dir = run_root / "_dev_shard"
        if not build_dev_shard(Path(cfg["corpus_dir"]), shard_dir,
                               int(cfg["dev_train_tokens"]), int(cfg["dev_train_offset"])):
            _harness_error("could not build the public dev shard (corpus missing/too small)")
            return 0
        out_dir = run_root / "dev_seed"
        ckpt, uid_dropped, observed_euid, note, stderr_txt = train_candidate(
            cfg, submission_py, shard_dir, out_dir)
        if ckpt is None:
            # distinguish an infra failure (import/traceback, no test ran) from a real
            # "recipe produced no checkpoint" — the former is refunded as harness_error.
            infra = any(s in (stderr_txt or "") for s in
                        ("ModuleNotFoundError", "ImportError", "Traceback (most recent call last)"))
            if infra and "train_for_budget missing" not in (stderr_txt or ""):
                _harness_error("dev training subprocess raised an infra error before producing a "
                               f"checkpoint: {(stderr_txt or '').strip().splitlines()[-1:] }")
                return 0
            _write_state(False, INV_NO_CHECKPOINT, None, 0.0, uid_dropped, observed_euid,
                         {"note": note})
            return 0
        try:
            ok, inv, bpb, detail = evaluate(cfg, submission_py, ckpt)
        except Exception as exc:
            _harness_error(f"dev evaluate() raised: {type(exc).__name__}: {exc}")
            return 0
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
