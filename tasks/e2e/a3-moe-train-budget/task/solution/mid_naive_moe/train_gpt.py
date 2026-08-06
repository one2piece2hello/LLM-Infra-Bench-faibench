# e2e-a3-moe-train-budget : REVIEWER-ONLY MID-RUNG control (a partially-correct MoE).
#
# 🔴 NOT SHIPPED. Same MoE MODEL as the oracle (same total params -> clears the capacity floor
# and the anti-spoof kit) but a NAIVE per-expert dispatch that runs EVERY expert on EVERY token
# and then masks. That is DENSE FLOPs (n_experts * expert_cost per token, not top_k), so it
# fits far fewer optimizer steps into the wall-clock budget; it also omits the load-balancing
# auxiliary loss, so the router tends to collapse. Result: it does beat the dense starter but
# lands well short of the oracle -> a reward strictly between 0 and 0.5.
#
# This is the discrimination probe: it proves the bounded reward grades PARTIAL MoE work
# instead of collapsing to pass/fail.
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.environ.get("NANOGPT_REPO", "/app/repo")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from model import CausalSelfAttention, LayerNorm, GPTConfig  # noqa: E402


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

    n_layer = 8
    n_head = 8
    n_embd = 512
    dropout = 0.0
    bias = False
    n_experts = 16
    top_k = 2
    ffn_mult = 4
    capacity_factor = 1.25       # unused by the naive dispatch (kept for cfg-dict parity)
    aux_loss_coef = 0.0          # NAIVE: no load-balancing loss (router can collapse)
    router_z_coef = 0.0

    micro_batch = 16
    grad_accum = 2
    learning_rate = 3e-3
    weight_decay = 0.1
    beta1, beta2 = 0.9, 0.95
    grad_clip = 1.0
    warmup_frac = 0.05
    min_lr_frac = 0.1
    save_every_sec = 20.0


class MoEMLP(nn.Module):
    """NAIVE dispatch: run every expert on every token, then combine with the
    top-k gate mask. Correct MoE math, but DENSE compute -> the sparsity is NOT
    realized as throughput. This is exactly the anti-pattern the task rewards
    replacing with a real sorted/grouped dispatch."""

    def __init__(self, cfg: "Hyperparameters"):
        super().__init__()
        self.d = cfg.n_embd
        self.hdim = cfg.ffn_mult * cfg.n_embd
        self.E = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(self.d, self.E, bias=False)
        self.w1 = nn.Parameter(torch.empty(self.E, self.d, self.hdim))
        self.w2 = nn.Parameter(torch.empty(self.E, self.hdim, self.d))
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.last_aux = torch.zeros(())
        self.last_z = torch.zeros(())

    def forward(self, x):
        B, T, d = x.shape
        N = B * T
        xf = x.reshape(N, d)
        logits = self.router(xf)
        probs = F.softmax(logits.float(), dim=-1).to(xf.dtype)      # [N, E]
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        # gate weight matrix [N, E]: nonzero only at the top_k experts
        gate = xf.new_zeros(N, self.E)
        gate.scatter_(1, topi, topv)
        self.last_aux = torch.zeros((), device=x.device)
        self.last_z = torch.zeros((), device=x.device)

        # NAIVE: every expert computes every token (DENSE FLOPs), then weight by gate.
        y = xf.new_zeros(N, d)
        for e in range(self.E):
            he = F.gelu(xf @ self.w1[e])          # [N, hdim]  -- ALL tokens
            oe = he @ self.w2[e]                  # [N, d]
            y = y + gate[:, e:e + 1] * oe
        return y.reshape(B, T, d)


class MoEBlock(nn.Module):
    def __init__(self, cfg, gptcfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(gptcfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.moe = MoEMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x


class MoEGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        gptcfg = GPTConfig(block_size=cfg.block_size, vocab_size=cfg.vocab_size,
                           n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                           dropout=cfg.dropout, bias=cfg.bias)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([MoEBlock(cfg, gptcfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def aux_losses(self):
        z = torch.zeros((), device=self.lm_head.weight.device)
        return z, z

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.h:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        return logits, None


class _VerifierAdapter(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, idx, targets=None):
        logits, loss = self.moe(idx, idx if targets is None else targets)
        return logits if targets is None else loss


def _cfg_dict(cfg):
    return dict(vocab_size=cfg.vocab_size, block_size=cfg.block_size,
                n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                dropout=0.0, bias=cfg.bias, n_experts=cfg.n_experts,
                top_k=cfg.top_k, ffn_mult=cfg.ffn_mult, capacity_factor=cfg.capacity_factor)


def _cfg_from_dict(d):
    c = Hyperparameters()
    for k, v in d.items():
        setattr(c, k, v)
    c.train_seq_len = d.get("block_size", c.block_size)
    return c


def load_model_for_verification(checkpoint_path, device):
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_dict(obj.get("config", {}))
    moe = MoEGPT(cfg)
    moe.load_state_dict(obj["model"], strict=True)
    return cfg, _VerifierAdapter(moe).to(device).eval()


def _save(model, cfg, out_ckpt):
    tmp = out_ckpt + ".tmp"
    torch.save({"model": model.state_dict(), "config": _cfg_dict(cfg)}, tmp)
    os.replace(tmp, out_ckpt)


class _MemmapLoader:
    def __init__(self, data_path, seq_len, device, seed):
        self.data = np.memmap(Path(data_path) / "train.bin", dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)

    def batch(self, n):
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=n)
        x = torch.from_numpy(np.stack([self.data[i:i + self.seq_len].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([self.data[i + 1:i + 1 + self.seq_len].astype(np.int64) for i in ix]))
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))


def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    model = MoEGPT(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[naive] MoE {n_params/1e6:.1f}M total params | DENSE per-expert loop dispatch (no throughput win)", flush=True)

    loader = _MemmapLoader(args.data_path, args.train_seq_len, device, args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), fused=True)
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
                _, ce = model(x, y)
            (ce / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        if step % 25 == 0:
            print(f"[naive] step {step} lr {lr:.2e} ce {ce.item():.4f} "
                  f"peakGB {torch.cuda.max_memory_allocated()/1e9:.2f} t {time.time()-start:.0f}s", flush=True)
        if time.time() - last_save >= args.save_every_sec:
            _save(model, args, args.out_ckpt)
            last_save = time.time()
    _save(model, args, args.out_ckpt)
    print(f"[naive] done: {step} steps in {time.time()-start:.0f}s -> {args.out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
