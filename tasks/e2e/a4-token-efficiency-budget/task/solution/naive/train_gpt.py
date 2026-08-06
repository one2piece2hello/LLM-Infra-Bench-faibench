# e2e-a4-token-efficiency-budget : the NAIVE floor control (reviewer-only).
# An untuned AdamW recipe: default-ish LR, no warmup, no cooldown. MEASURED on NVIDIA H20 at
# the 12M-token probe budget: 7.8369 bits/token -> reward 0.9887 (< the 1.0 anchor).
#
# THE TASK. The grader gives you a FIXED NUMBER OF TRAINING TOKENS and scores the held-out
# val_bpb of the checkpoint your recipe produces. Wall-clock is NOT the budget: making the
# step faster does not let you consume more tokens, so throughput buys you NOTHING. The only
# way to a lower loss is to extract MORE LEARNING PER TOKEN -- real sample efficiency.
#
# THIS FILE IS BOTH (a) the reviewer's strong baseline (the 1.0 anchor you must beat) and
# (b) a starting point you may rewrite entirely. It is NOT a contract. The tuned-AdamW recipe
# below was selected as the BEST of a batch-size/learning-rate sweep at this token budget, so
# re-tuning those knobs will not move your score -- you have to change what the training
# ALGORITHM is (optimizer / architecture / data schedule).
#
# THE ONLY CONTRACT (both required):
#   train_for_budget(seed, out_dir, budget_seconds) -> writes <out_dir>/model_ckpt.pt
#   load_model_for_verification(checkpoint_path, device) -> nn.Module or (args, model)
# with model(x) -> logits (B,T,V) position-dependent, and model(x, y) -> scalar loss equal to
# F.cross_entropy(logits.view(-1,V), y.view(-1)) on those logits.
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
from model import GPT, GPTConfig  # noqa: E402  (from /app/repo/model.py -- fully editable)


class Hyperparameters:
    """Starting point. Nothing here is required by scoring."""
    # The grader points DATA_PATH at YOUR TOKEN BUDGET: a shard holding exactly
    # TOKEN_BUDGET tokens. It is the only readable token source during scoring.
    data_path = os.environ.get("DATA_PATH", "/data/datasets/wikitext_bpe8k")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tokenizers/wikitext_bpe8k.model")
    seed = int(os.environ.get("SEED", "1337"))
    token_budget = int(os.environ.get("TOKEN_BUDGET", "24000000"))
    max_params = int(os.environ.get("MAX_PARAMS", "45000000"))

    vocab_size = 8192
    train_seq_len = 512          # the verifier reads this as seq_len
    block_size = 512
    val_batch_size = 32_768

    # ~34M-parameter model, comfortably inside the parameter cap the grader re-counts.
    n_layer = 8
    n_head = 8
    n_embd = 512
    dropout = 0.0
    bias = False

    # TUNED AdamW at this token budget (best of a micro-batch x LR sweep).
    micro_batch = 8
    grad_accum = 1
    learning_rate = 3.0e-4      # untuned
    weight_decay = 0.1
    beta1, beta2 = 0.9, 0.95
    grad_clip = 1.0
    warmup_frac = 0.0            # no warmup
    cooldown_frac = 0.001        # no cooldown
    min_lr_frac = 0.02
    save_every_sec = 20.0        # save periodically: the grader may stop you at any moment


class TokenStream:
    """Sequential single-epoch reader over the budgeted shard.

    Reads the shard start-to-end exactly once, so the recipe consumes its token budget and
    no token is repeated (repetition invites memorisation, which the held-out eval punishes).
    The loader implementation is entirely yours to change -- order, packing, curriculum and
    sequence length are all levers."""

    def __init__(self, data_path: str, seq_len: int, device):
        self.data = np.memmap(Path(data_path) / "train.bin", dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.device = device
        self.pos = 0
        self.tokens_served = 0
        self.exhausted = False

    def batch(self, n: int):
        need = n * self.seq_len + 1
        if self.pos + need >= len(self.data):
            self.pos = 0
            self.exhausted = True          # one full pass over the budget completed
        blk = np.asarray(self.data[self.pos:self.pos + need], dtype=np.int64)
        self.pos += n * self.seq_len
        self.tokens_served += n * self.seq_len
        x = torch.from_numpy(blk[:-1].reshape(n, self.seq_len))
        y = torch.from_numpy(blk[1:].reshape(n, self.seq_len))
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))


class _VerifierAdapter(nn.Module):
    """nanoGPT's GPT.forward(idx, targets) returns (logits, loss) and only computes full
    logits when targets are supplied; adapt it to the verifier's forward contract."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx, targets=None):
        logits, loss = self.gpt(idx, idx if targets is None else targets)
        return logits if targets is None else loss


def _build(args: Hyperparameters, device) -> GPT:
    cfg = GPTConfig(block_size=args.block_size, vocab_size=args.vocab_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=args.dropout, bias=args.bias)
    return GPT(cfg).to(device)


def _save(model: GPT, cfg: GPTConfig, out_ckpt: Path) -> None:
    m = getattr(model, "_orig_mod", model)
    tmp = str(out_ckpt) + ".tmp"
    torch.save({"model": m.state_dict(),
                "config": dict(block_size=cfg.block_size, vocab_size=cfg.vocab_size,
                               n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                               dropout=0.0, bias=cfg.bias)}, tmp)
    os.replace(tmp, out_ckpt)      # atomic: a hard kill never leaves a half-written ckpt


def load_model_for_verification(checkpoint_path, device):
    args = Hyperparameters()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ck = obj.get("config", {})
    cfg = GPTConfig(**ck) if ck else GPTConfig(
        block_size=args.block_size, vocab_size=args.vocab_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=0.0, bias=args.bias)
    gpt = GPT(cfg)
    sd = obj["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    gpt.load_state_dict(sd, strict=True)
    args.train_seq_len = cfg.block_size
    args.vocab_size = cfg.vocab_size
    return args, _VerifierAdapter(gpt).to(device).eval()


def train_for_budget(seed: int, out_dir, budget_seconds: float) -> None:
    """Train on the token budget the grader made available at DATA_PATH.

    `budget_seconds` is only a FEASIBILITY GUARD (generous); the real budget is the tokens.
    """
    args = Hyperparameters()
    args.seed = int(seed)
    out_ckpt = Path(out_dir) / "model_ckpt.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    model = _build(args, device)
    cfg = model.config
    stream = TokenStream(args.data_path, args.train_seq_len, device)
    # The shard IS the budget: one pass over it consumes exactly the token budget.
    budget_tokens = int(stream.data.shape[0])
    tokens_per_step = args.micro_batch * args.grad_accum * args.train_seq_len
    total_steps = max(1, budget_tokens // tokens_per_step)
    print(f"[train] seed={seed} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"budget={budget_tokens/1e6:.2f}M tokens -> {total_steps} steps "
          f"(micro={args.micro_batch} accum={args.grad_accum} seq={args.train_seq_len})",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(args.beta1, args.beta2),
                            weight_decay=args.weight_decay, fused=True)
    warm = max(1, int(args.warmup_frac * total_steps))
    cool = max(1, int(args.cooldown_frac * total_steps))
    t0 = last_save = time.time()
    guard = float(budget_seconds) * 0.94 if budget_seconds else float("inf")
    model.train()
    for step in range(total_steps):
        if time.time() - t0 > guard:                   # never get hard-killed mid-save
            break
        if step < warm:
            f = (step + 1) / warm
        elif step > total_steps - cool:
            f = max(args.min_lr_frac, (total_steps - step) / cool)
        else:
            f = 1.0
        for g in opt.param_groups:
            g["lr"] = args.learning_rate * f
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = stream.batch(args.micro_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % 200 == 0:
            print(f"[train] step {step}/{total_steps} lr {args.learning_rate*f:.2e} "
                  f"loss {float(loss):.4f} tok {stream.tokens_served/1e6:.1f}M "
                  f"t {time.time()-t0:.0f}s", flush=True)
        if time.time() - last_save >= args.save_every_sec:
            _save(model, cfg, out_ckpt)
            last_save = time.time()
    _save(model, cfg, out_ckpt)
    print(f"[train] done: {stream.tokens_served/1e6:.2f}M tokens consumed in "
          f"{time.time()-t0:.0f}s -> {out_ckpt}", flush=True)


if __name__ == "__main__":
    train_for_budget(int(os.environ.get("SEED", "1337")),
                     os.environ.get("OUT_DIR", "/app/submission"),
                     float(os.environ.get("MAX_WALLCLOCK_SEC", "900")))
