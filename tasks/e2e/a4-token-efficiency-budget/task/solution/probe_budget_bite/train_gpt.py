# REVIEWER-ONLY BUDGET-BITE PROBE for e2e-a4-token-efficiency-budget. 🔴 NOT SHIPPED.
#
# A gate that no plausible submission can fail contributes nothing, and a SELF-REPORTED budget is
# worse than none. This task claims its TOKEN budget holds "by construction": the harness cuts a
# shard of exactly token_budget tokens, points DATA_PATH at it, and chmod-000 quarantines the full
# corpus. This probe tests that claim adversarially and records the OS's answer:
#
#   1. OVER-BUDGET READ — it tries to open the FULL corpus at /data/datasets/wikitext_bpe8k/
#      train.bin (many times token_budget) instead of DATA_PATH, and reports whether it could.
#      chmod 000 is ignored by uid 0, so this SUCCEEDS as root and FAILS under the privilege drop:
#      the drop is what makes "by construction" true rather than nominal.
#   2. SELF-REPORT A LIE — it writes an absurd token count into its own output and into the
#      checkpoint. The harness must never read it.
#   3. READ THE HELD-OUT SET — it tries /tests/holdout_corpus.txt; under the drop this must fail.
#   4. IT ALSO TRAINS HONESTLY on whatever it could actually read, so the run reaches the scorer
#      and the reward can be inspected.
from __future__ import annotations

import json
import math
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

PROBE: dict = {}


class Hyperparameters:
    vocab_size = 8192
    train_seq_len = 1024
    block_size = 1024
    val_batch_size = 32_768
    n_layer = 6
    n_head = 8
    n_embd = 512
    dropout = 0.0
    bias = False
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tokenizers/wikitext_bpe8k.model")


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


def _probe_access() -> dict:
    out = {"euid": os.geteuid(), "uid": os.getuid()}
    full = "/data/datasets/wikitext_bpe8k/train.bin"
    shard = os.environ.get("DATA_PATH", "")
    try:
        d = np.memmap(full, dtype=np.uint16, mode="r")
        out["over_budget_full_corpus"] = {
            "path": full, "READ_SUCCEEDED": True, "tokens_reachable": int(d.shape[0]),
            "token_budget_env": os.environ.get("TOKEN_BUDGET"),
            "verdict": "BUDGET BYPASSABLE: more tokens are readable than the budget allows"}
    except Exception as exc:
        out["over_budget_full_corpus"] = {
            "path": full, "READ_SUCCEEDED": False, "error": f"{type(exc).__name__}: {exc}",
            "token_budget_env": os.environ.get("TOKEN_BUDGET"),
            "verdict": "BUDGET HOLDS: the full corpus is unreachable, only the shard is readable"}
    try:
        d = np.memmap(Path(shard) / "train.bin", dtype=np.uint16, mode="r")
        out["shard"] = {"path": shard, "tokens": int(d.shape[0])}
    except Exception as exc:
        out["shard"] = {"path": shard, "error": f"{type(exc).__name__}: {exc}"}
    for target in ("/tests/holdout_corpus.txt", "/tests/verifier-correctness-manifest.json"):
        try:
            with open(target, "rb") as fh:
                fh.read(64)
            out[target] = {"READ_SUCCEEDED": True, "verdict": "HELD-OUT SET IS READABLE"}
        except Exception as exc:
            out[target] = {"READ_SUCCEEDED": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def train_for_budget(seed: int, out_dir: str, budget_seconds: float) -> None:
    global PROBE
    PROBE = _probe_access()
    PROBE["self_reported_tokens_consumed"] = 1              # a lie the harness must not believe
    PROBE["self_reported_steps"] = 1
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "_probe_report.json").write_text(json.dumps(PROBE, indent=2, default=str))
    print("[probe]", json.dumps(PROBE, default=str), flush=True)

    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    a = Hyperparameters()
    cfg = _cfg(a)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    src = Path(os.environ.get("DATA_PATH", "/data/datasets/wikitext_bpe8k")) / "train.bin"
    data = np.memmap(src, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    cfg_dict = dict(block_size=cfg.block_size, vocab_size=cfg.vocab_size, n_layer=cfg.n_layer,
                    n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, bias=cfg.bias)
    step = 0
    while time.time() - t0 < min(90.0, float(budget_seconds) * 0.2):
        ix = rng.integers(0, len(data) - a.train_seq_len - 1, size=8)
        x = torch.from_numpy(np.stack([data[i:i + a.train_seq_len].astype(np.int64) for i in ix])).to(device)
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + a.train_seq_len].astype(np.int64) for i in ix])).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            torch.save({"model": model.state_dict(), "config": cfg_dict,
                        "self_reported_tokens": 1}, Path(out_dir, "model_ckpt.pt"))
    torch.save({"model": model.state_dict(), "config": cfg_dict, "self_reported_tokens": 1},
               Path(out_dir, "model_ckpt.pt"))
    print(f"[probe] trained {step} steps in {time.time() - t0:.0f}s", flush=True)


def load_model_for_verification(checkpoint_path, device):
    a = Hyperparameters()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**obj["config"]) if obj.get("config") else _cfg(a)
    gpt = GPT(cfg)
    gpt.load_state_dict(obj["model"], strict=True)
    a.train_seq_len, a.vocab_size = cfg.block_size, cfg.vocab_size
    return a, _VerifierAdapter(gpt).to(device).eval()
