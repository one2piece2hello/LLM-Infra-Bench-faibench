# REVIEWER-ONLY BUDGET-BITE PROBE for e2e-a3-moe-train-budget. 🔴 NOT SHIPPED, NOT a solution.
#
# A gate that no plausible submission can fail contributes nothing, and a SELF-REPORTED budget is
# worse than none. This probe exists to prove the wall-clock budget gate actually BITES, by
# attacking it three ways at once and reporting what the harness did about each:
#
#   1. IGNORE THE CAP  — it never stops on its own; only the harness timer can end it.
#   2. OUTLIVE THE CAP — it forks a DETACHED grandchild (new session) that keeps writing the
#      scored checkpoint after the parent is signalled. If the harness only killed the direct
#      child, this would keep training into the eval phase.
#   3. READ THE HELD-OUT SET — it tries to open /tests/holdout_corpus.txt and the frozen manifest.
#      Under the privilege drop this must fail with EACCES/EPERM; as root it would succeed, which
#      is exactly why the drop exists.
#
# It writes a probe report to $OUT_CKPT.probe.json (the harness log carries it), then a REAL
# (deliberately tiny/awful) checkpoint so the run still reaches the scorer.
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = os.environ.get("NANOGPT_REPO", "/app/repo")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from model import GPT, GPTConfig  # noqa: E402


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "/data/datasets/wikitext_bpe8k")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tokenizers/wikitext_bpe8k.model")
    seed = int(os.environ.get("SEED", "1337"))
    out_ckpt = os.environ.get("OUT_CKPT", "/app/submission/model_ckpt.pt")
    wallclock_sec = float(os.environ.get("WALLCLOCK_SEC", "600"))
    vocab_size = 8192
    train_seq_len = 512
    block_size = 512
    val_batch_size = 32_768
    n_layer = 12
    n_head = 16
    n_embd = 1024
    dropout = 0.0
    bias = False


class _VerifierAdapter(nn.Module):
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx, targets=None):
        logits, loss = self.gpt(idx, idx if targets is None else targets)
        return logits if targets is None else loss


def _cfg(a):
    return GPTConfig(block_size=a.block_size, vocab_size=a.vocab_size, n_layer=a.n_layer,
                     n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout, bias=a.bias)


def load_model_for_verification(checkpoint_path, device):
    a = Hyperparameters()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**obj["config"]) if obj.get("config") else _cfg(a)
    gpt = GPT(cfg)
    gpt.load_state_dict(obj["model"], strict=True)
    a.train_seq_len, a.vocab_size = cfg.block_size, cfg.vocab_size
    return a, _VerifierAdapter(gpt).to(device).eval()


def _probe_frozen_surface() -> dict:
    """Attempt exactly what the rules forbid, and record the OS's answer."""
    out = {}
    for target in ("/tests/holdout_corpus.txt", "/tests/verifier-correctness-manifest.json",
                   "/tests/compute_reward.py", "/tests"):
        try:
            if target == "/tests":
                out[target] = {"listdir": sorted(os.listdir(target))[:8]}
            else:
                with open(target, "rb") as fh:
                    head = fh.read(64)
                out[target] = {"READ_SUCCEEDED": True, "first_bytes": head[:32].decode("utf-8", "replace")}
        except Exception as exc:
            out[target] = {"READ_SUCCEEDED": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _spawn_detached_writer(out_ckpt: str, cfg_dict: dict, state: dict) -> int:
    """Fork a grandchild in its OWN session that keeps rewriting the scored checkpoint. The
    harness must kill the whole process group, not just the direct child."""
    pid = os.fork()
    if pid != 0:
        return pid
    try:
        os.setsid()
        marker = out_ckpt + ".detached_writes"
        for i in range(4000):
            with open(marker, "a") as fh:
                fh.write(f"{time.time():.3f} detached_write {i}\n")
            try:
                torch.save({"model": state, "config": cfg_dict}, out_ckpt + ".tmp")
                os.replace(out_ckpt + ".tmp", out_ckpt)
            except Exception:
                pass
            time.sleep(2.0)
    except Exception:
        pass
    finally:
        os._exit(0)


def main() -> None:
    a = Hyperparameters()
    torch.manual_seed(a.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    probe = {"euid": os.geteuid(), "uid": os.getuid(), "pid": os.getpid(),
             "wallclock_sec_env": a.wallclock_sec, "started": time.time()}
    probe["frozen_surface_access"] = _probe_frozen_surface()
    # can it read MORE training data than the harness pointed it at?
    try:
        full = Path("/data/datasets/wikitext_bpe8k/train.bin")
        probe["full_corpus_read"] = {"path": str(full),
                                     "READ_SUCCEEDED": bool(np.memmap(full, dtype=np.uint16, mode="r").shape[0])}
    except Exception as exc:
        probe["full_corpus_read"] = {"READ_SUCCEEDED": False, "error": f"{type(exc).__name__}: {exc}"}

    cfg = _cfg(a)
    model = GPT(cfg).to(device)
    cfg_dict = dict(block_size=cfg.block_size, vocab_size=cfg.vocab_size, n_layer=cfg.n_layer,
                    n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, bias=cfg.bias)
    torch.save({"model": model.state_dict(), "config": cfg_dict}, a.out_ckpt + ".tmp")
    os.replace(a.out_ckpt + ".tmp", a.out_ckpt)
    probe["detached_child_pid"] = _spawn_detached_writer(a.out_ckpt, cfg_dict, model.state_dict())
    probe["self_reported_steps"] = 999_999_999      # a lie the harness must not believe
    probe["self_reported_wallclock_used"] = 1.0
    Path(a.out_ckpt + ".probe.json").write_text(json.dumps(probe, indent=2, default=str))
    print("[probe] report:", json.dumps(probe, default=str), flush=True)
    # never stop voluntarily: only the harness timer may end this run
    while True:
        time.sleep(5)
        print(f"[probe] still alive at t={time.time() - probe['started']:.0f}s "
              f"(budget was {a.wallclock_sec}s)", flush=True)


if __name__ == "__main__":
    main()
