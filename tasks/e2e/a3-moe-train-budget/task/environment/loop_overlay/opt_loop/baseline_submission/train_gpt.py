# e2e-a3-moe-train-budget — the provided STARTING POINT (fully editable, NOT a contract).
#
# This is a plain DENSE nanoGPT sized to clear the task's hard total-parameter FLOOR, trained
# with a straightforward AdamW recipe until the harness's wall-clock timer stops it. It is a
# complete, working, honest baseline: real data loading, bf16 autocast, warmup + cosine decay,
# gradient clipping, and an atomic periodic checkpoint save so a timer kill always leaves a
# valid artifact. It satisfies the verifier's entry contract as-is.
#
# It is also DELIBERATELY the thing you have to beat. Because it is dense, every token pays
# the full ~2*P_total FLOPs, so it fits comparatively few optimizer steps into the budget and
# converges to a mediocre held-out bits-per-byte. Submitting it unchanged scores 0.
#
# You may change anything: the model, the router, the dispatch, the optimizer, the schedule,
# the data loading, the precision — everything under /app/repo (nanoGPT) and /app/submission.
#
# VERIFIER CONTRACT (enforced): the grader loads the checkpoint via
# load_model_for_verification(checkpoint_path, device) and requires a model whose
#   * model(input_ids)             -> float logits (batch, seq, vocab), position-dependent
#   * model(input_ids, target_ids) -> scalar loss == CE(logits.view(-1,V), targets.view(-1))
# i.e. the PURE cross-entropy of those logits. If you add an auxiliary training objective
# (for example a load-balancing loss), keep it inside your training loop — do NOT fold it into
# the loss this forward returns. An OOD probe enforces loss(real_text) < 0.7 * loss(arbitrary
# ids), and the model must clear the total-parameter floor.
from __future__ import annotations

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
from model import GPT, GPTConfig  # noqa: E402  (from /app/repo/model.py)


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

    # DENSE model sized to clear the 150M floor: n_embd=1024,n_layer=12 nanoGPT
    # transformer ~= 12*(12*1024^2) = 151.0M + tied wte 8.4M + wpe 0.5M ~= 160M.
    n_layer = 12
    n_head = 16
    n_embd = 1024
    dropout = 0.0
    bias = False

    # SAME recipe as the strong MoE baseline (fair comparison; only dense-vs-sparse differs)
    micro_batch = 16
    grad_accum = 2
    learning_rate = 3e-3
    weight_decay = 0.1
    beta1, beta2 = 0.9, 0.95
    grad_clip = 1.0
    warmup_frac = 0.05
    min_lr_frac = 0.1
    save_every_sec = 20.0


class _VerifierAdapter(nn.Module):
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx, targets=None):
        logits, loss = self.gpt(idx, idx if targets is None else targets)
        return logits if targets is None else loss


def _cfg(args: Hyperparameters) -> GPTConfig:
    return GPTConfig(block_size=args.block_size, vocab_size=args.vocab_size,
                     n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                     dropout=args.dropout, bias=args.bias)


def load_model_for_verification(checkpoint_path, device):
    args = Hyperparameters()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_kwargs = obj.get("config", {})
    cfg = GPTConfig(**cfg_kwargs) if cfg_kwargs else _cfg(args)
    gpt = GPT(cfg)
    gpt.load_state_dict(obj["model"], strict=True)
    args.train_seq_len = cfg.block_size
    args.vocab_size = cfg.vocab_size
    return args, _VerifierAdapter(gpt).to(device).eval()


def _save(model: GPT, cfg: GPTConfig, out_ckpt: str) -> None:
    tmp = out_ckpt + ".tmp"
    torch.save({"model": model.state_dict(),
                "config": dict(block_size=cfg.block_size, vocab_size=cfg.vocab_size,
                               n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                               dropout=0.0, bias=cfg.bias)}, tmp)
    os.replace(tmp, out_ckpt)


class _MemmapLoader:
    def __init__(self, data_path: str, seq_len: int, device: torch.device, seed: int):
        self.data = np.memmap(Path(data_path) / "train.bin", dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)

    def batch(self, n: int):
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=n)
        x = torch.from_numpy(np.stack([self.data[i:i + self.seq_len].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([self.data[i + 1:i + 1 + self.seq_len].astype(np.int64) for i in ix]))
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))


def main() -> None:
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    cfg = _cfg(args)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] DENSE {n_params/1e6:.1f}M params | layers={args.n_layer} "
          f"d={args.n_embd} | micro={args.micro_batch} accum={args.grad_accum} "
          f"seq={args.train_seq_len}", flush=True)

    loader = _MemmapLoader(args.data_path, args.train_seq_len, device, args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
                            fused=True)

    start = time.time()
    deadline = start + args.wallclock_sec * 0.94
    est_total_steps = 6000
    warmup = max(1, int(args.warmup_frac * est_total_steps))
    last_save = start
    step = 0
    model.train()
    while time.time() < deadline:
        lr = (args.learning_rate * (step + 1) / warmup if step < warmup
              else args.learning_rate * (args.min_lr_frac + (1 - args.min_lr_frac) *
                   0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, est_total_steps - warmup))))))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = loader.batch(args.micro_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        if step % 25 == 0:
            print(f"[train] step {step} lr {lr:.2e} loss {loss.item():.4f} "
                  f"peakGB {torch.cuda.max_memory_allocated()/1e9:.2f} t {time.time()-start:.0f}s", flush=True)
        if time.time() - last_save >= args.save_every_sec:
            _save(model, cfg, args.out_ckpt)
            last_save = time.time()
    _save(model, cfg, args.out_ckpt)
    print(f"[train] done: {step} steps in {time.time()-start:.0f}s -> {args.out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
