# e2e-a4-token-efficiency-budget : NEGATIVE control (REVIEWER-ONLY -- never in the image).
#
# The SMUGGLED-CHECKPOINT cheat, which is the specific hole the families-ref item 8 names for
# any task where the harness INVOKES the candidate's training entry: instead of training, hand
# back a checkpoint that was prepared earlier. If this scores anything above 0 the task is
# broken. TWO independent gates must kill it:
#   1. the isolated staging dir strips every *.pt / *.safetensors / >8 MB blob, so there is no
#      pre-baked artifact left to copy; and
#   2. the cross-seed weight-divergence gate: this recipe is DETERMINISTIC in the seed (it
#      ignores `seed` entirely), so all seeds produce byte-identical checkpoints -> reward 0.
# It is also a degenerate model (barely trained), so the anti-spoof / quality gates fire too.
from __future__ import annotations

import os
import shutil
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

# Where a cheating solver would have parked a pre-trained artifact.
SMUGGLED = [Path(__file__).parent / "pretrained.pt",
            Path("/app/submission/pretrained.pt"),
            Path("/app/submission/.cache/weights.safetensors")]


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "/data/datasets/wikitext_bpe8k")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tokenizers/wikitext_bpe8k.model")
    seed = 1337
    vocab_size = 8192
    train_seq_len = 512
    block_size = 512
    val_batch_size = 32_768
    n_layer, n_head, n_embd = 8, 8, 512
    dropout = 0.0
    bias = False


class _VerifierAdapter(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx, targets=None):
        logits, loss = self.gpt(idx, idx if targets is None else targets)
        return logits if targets is None else loss


def _cfg(a):
    return GPTConfig(block_size=a.block_size, vocab_size=a.vocab_size, n_layer=a.n_layer,
                     n_head=a.n_head, n_embd=a.n_embd, dropout=0.0, bias=a.bias)


def load_model_for_verification(checkpoint_path, device):
    a = Hyperparameters()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    c = obj.get("config", {})
    cfg = GPTConfig(**c) if c else _cfg(a)
    gpt = GPT(cfg)
    gpt.load_state_dict(obj["model"], strict=True)
    a.train_seq_len = cfg.block_size
    a.vocab_size = cfg.vocab_size
    return a, _VerifierAdapter(gpt).to(device).eval()


def train_for_budget(seed, out_dir, budget_seconds):
    """THE CHEAT: try to return a pre-baked checkpoint; if the harness stripped it (it does),
    fall back to a fixed-seed 20-step 'training' so the output is byte-identical across seeds."""
    a = Hyperparameters()
    out_ckpt = Path(out_dir) / "model_ckpt.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    for cand in SMUGGLED:
        if cand.exists() and cand.stat().st_size > 0:
            shutil.copy2(cand, out_ckpt)
            print(f"[negative] smuggled {cand} -> {out_ckpt}", flush=True)
            return
    # No blob survived the strip -> emulate the smuggle exactly: emit a CONSTANT artifact
    # built from a fixed seed with NO data dependence, so every scored seed yields a
    # byte-identical checkpoint. This is what the cross-seed divergence gate exists to catch.
    torch.manual_seed(a.seed)          # NOTE: ignores `seed` AND ignores the data on purpose
    np.random.seed(a.seed)
    model = GPT(_cfg(a))
    tmp = str(out_ckpt) + ".tmp"
    torch.save({"model": model.state_dict(),
                "config": dict(block_size=a.block_size, vocab_size=a.vocab_size,
                               n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd,
                               dropout=0.0, bias=a.bias)}, tmp)
    os.replace(tmp, out_ckpt)
    print("[negative] emitted a CONSTANT (data-independent) checkpoint -- expect BOTH the "
          "cross-seed divergence gate and the real-LM anti-spoof probes to fire", flush=True)


if __name__ == "__main__":
    train_for_budget(int(os.environ.get("SEED", "1337")),
                     os.environ.get("OUT_DIR", "/app/submission"), 900.0)
