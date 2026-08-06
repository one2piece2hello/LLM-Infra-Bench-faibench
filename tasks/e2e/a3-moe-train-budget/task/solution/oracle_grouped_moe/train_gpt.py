# e2e-a3-moe-train-budget : REVIEWER-ONLY ORACLE (the ref_speedup numerator's denominator).
#
# 🔴 NOT SHIPPED. This file is the answer: a real single-card sparse Mixture-of-Experts recipe
# (sorted grouped-GEMM token dispatch + a Switch-style load-balancing auxiliary loss + a tuned
# top-k / n_experts / capacity frontier) that clears the total-parameter FLOOR and converts the
# sparsity into real throughput, so it fits far more optimizer steps into the fixed wall-clock
# budget than the dense starter and reaches a much lower held-out val_bpb.
#
# It defines the reward's 0.5 point:  ref_speedup = baseline_bpb / oracle_val_bpb, a FROZEN
# constant calibrated on NVIDIA H20 and carried in the frozen manifest. The oracle is never present
# in the task image and is never executed at scoring time — only the constant is read.
#
# The SHIPPED starting point is the dense-at-floor recipe in environment/submission/, which is
# what `baseline_bpb` is calibrated against.
#
# VERIFIER CONTRACT (enforced): model(x) -> float logits (B,T,V), position-dependent;
# model(x,y) -> scalar loss == CE(logits.view(-1,V), y.view(-1)), i.e. the PURE cross-entropy of
# those logits — the MoE auxiliary loss is a TRAINING-time side channel and must NOT be folded
# into the forward-loss. An OOD probe enforces loss(real) < 0.7 * loss(arbitrary ids), and the
# model must clear the total-parameter floor.
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

# nanoGPT (the baked, editable system) lives at /app/repo. We reuse its attention
# + LayerNorm + config so the repo is load-bearing (you may edit those too).
REPO = os.environ.get("NANOGPT_REPO", "/app/repo")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from model import CausalSelfAttention, LayerNorm, GPTConfig  # noqa: E402  (from /app/repo/model.py)


# ---- Hyperparameters (starting point; nothing here is required by scoring) ----
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "/data/datasets/wikitext_bpe8k")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tokenizers/wikitext_bpe8k.model")
    seed = int(os.environ.get("SEED", "1337"))
    out_ckpt = os.environ.get("OUT_CKPT", "/app/submission/model_ckpt.pt")
    wallclock_sec = float(os.environ.get("WALLCLOCK_SEC", "600"))

    vocab_size = 8192
    train_seq_len = 512         # verifier reads this as seq_len
    block_size = 512
    val_batch_size = 32_768     # verifier eval batch (tokens)

    # MoE model: many small experts, few active per token -> P_floor total
    # capacity at ACTIVE-param compute cost. These sizes put total params well
    # above the floor while keeping active params (hence FLOPs/token) small.
    n_layer = 8
    n_head = 8
    n_embd = 512
    dropout = 0.0
    bias = False
    # --- MoE FFN config (the lever) ---
    n_experts = 16              # experts per MoE FFN
    top_k = 2                   # experts active per token
    ffn_mult = 4                # expert hidden = ffn_mult * n_embd
    capacity_factor = 1.25      # per-expert buffer = capacity_factor * top_k * tokens / n_experts
    aux_loss_coef = 0.01        # load-balancing auxiliary loss weight (Switch-style)
    router_z_coef = 1e-3        # router z-loss (keeps logits from blowing up)

    # training recipe
    micro_batch = 16            # sequences per forward
    grad_accum = 2              # effective batch = micro_batch * grad_accum
    learning_rate = 3e-3
    weight_decay = 0.1
    beta1, beta2 = 0.9, 0.95
    grad_clip = 1.0
    warmup_frac = 0.05
    min_lr_frac = 0.1
    save_every_sec = 20.0       # periodic checkpoint so a timer-kill keeps the latest


# ============================ the single-card MoE ============================
class MoEMLP(nn.Module):
    """A single-card top-k Mixture-of-Experts feed-forward layer.

    STRONG-BASELINE dispatch: sort the (token, expert) assignments by expert, drop
    per-expert overflow beyond `capacity`, and compute all experts as a BATCHED
    matmul (torch.bmm over [n_experts, capacity, d]) so each expert only touches
    its routed tokens. This realizes the MoE sparsity as real single-card
    throughput (grouped-GEMM style). The load-balancing aux loss + router z-loss
    are exposed via `self.last_aux` / `self.last_z` for the training loop to add;
    they are NOT part of the forward output (the verifier checks pure CE).
    """

    def __init__(self, cfg: "Hyperparameters"):
        super().__init__()
        self.d = cfg.n_embd
        self.hdim = cfg.ffn_mult * cfg.n_embd
        self.E = cfg.n_experts
        self.top_k = cfg.top_k
        self.capacity_factor = cfg.capacity_factor
        self.router = nn.Linear(self.d, self.E, bias=False)
        # expert weights as batched parameters [E, in, out]
        self.w1 = nn.Parameter(torch.empty(self.E, self.d, self.hdim))
        self.w2 = nn.Parameter(torch.empty(self.E, self.hdim, self.d))
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.last_aux = torch.zeros(())
        self.last_z = torch.zeros(())

    def forward(self, x):
        # x: [B, T, d]
        B, T, d = x.shape
        N = B * T
        xf = x.reshape(N, d)
        logits = self.router(xf)                       # [N, E]
        probs = F.softmax(logits.float(), dim=-1).to(xf.dtype)   # [N, E]

        # load-balancing aux (Switch): f_i = fraction of tokens whose TOP-1 is i;
        # P_i = mean router prob to i; aux = E * sum_i f_i * P_i.
        if self.training:
            top1 = probs.argmax(dim=-1)
            f = torch.zeros(self.E, device=x.device, dtype=probs.dtype)
            f.scatter_add_(0, top1, torch.ones_like(top1, dtype=probs.dtype))
            f = f / max(N, 1)
            P = probs.mean(dim=0)
            self.last_aux = self.E * torch.sum(f * P)
            self.last_z = torch.mean(torch.logsumexp(logits.float(), dim=-1) ** 2)
        else:
            self.last_aux = torch.zeros((), device=x.device)
            self.last_z = torch.zeros((), device=x.device)

        topv, topi = probs.topk(self.top_k, dim=-1)    # [N, k]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)

        capacity = max(1, int(self.capacity_factor * self.top_k * N / self.E))
        flat_e = topi.reshape(-1)                      # [N*k]
        flat_g = topv.reshape(-1)                      # [N*k]
        flat_src = torch.arange(N, device=x.device).repeat_interleave(self.top_k)  # token id

        # sort assignments by expert -> contiguous per-expert groups
        sort_e, order = torch.sort(flat_e)
        src_sorted = flat_src[order]
        g_sorted = flat_g[order]
        counts = torch.bincount(flat_e, minlength=self.E)     # [E]
        starts = torch.cumsum(counts, 0) - counts             # start index per expert
        pos_within = torch.arange(flat_e.numel(), device=x.device) - starts[sort_e]
        keep = pos_within < capacity                          # drop per-expert overflow

        e_sel = sort_e[keep]
        slot_sel = pos_within[keep]
        tok_sel = src_sorted[keep]
        g_sel = g_sorted[keep]

        # scatter kept tokens into a padded [E, capacity, d] dispatch buffer
        disp = xf.new_zeros(self.E, capacity, d)
        disp[e_sel, slot_sel] = xf[tok_sel]

        # batched expert compute (grouped GEMM): [E,C,d] @ [E,d,h] -> gelu -> @ [E,h,d]
        hpre = torch.bmm(disp, self.w1)
        hact = F.gelu(hpre)
        out = torch.bmm(hact, self.w2)                        # [E, C, d]

        # combine back to tokens, weighted by the (renormalized) gate
        y = xf.new_zeros(N, d)
        contrib = out[e_sel, slot_sel] * g_sel.unsqueeze(-1)
        y.index_add_(0, tok_sel, contrib)
        return y.reshape(B, T, d)


class MoEBlock(nn.Module):
    def __init__(self, cfg: "Hyperparameters", gptcfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(gptcfg)        # nanoGPT attention (repo is load-bearing)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.moe = MoEMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x


class MoEGPT(nn.Module):
    def __init__(self, cfg: "Hyperparameters"):
        super().__init__()
        self.cfg = cfg
        gptcfg = GPTConfig(block_size=cfg.block_size, vocab_size=cfg.vocab_size,
                           n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                           dropout=cfg.dropout, bias=cfg.bias)
        self.gptcfg = gptcfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([MoEBlock(cfg, gptcfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight          # weight tying (nanoGPT)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def aux_losses(self):
        aux = torch.zeros((), device=self.lm_head.weight.device)
        z = torch.zeros((), device=self.lm_head.weight.device)
        for blk in self.h:
            aux = aux + blk.moe.last_aux
            z = z + blk.moe.last_z
        n = max(len(self.h), 1)
        return aux / n, z / n

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
            return logits, loss          # PURE CE (aux is a side channel via aux_losses())
        return logits, None


# ---- Verifier hook: reconstruct + adapt to the forward(x)/forward(x,y) contract ----
class _VerifierAdapter(nn.Module):
    def __init__(self, moe: MoEGPT):
        super().__init__()
        self.moe = moe

    def forward(self, idx, targets=None):
        logits, loss = self.moe(idx, idx if targets is None else targets)
        return logits if targets is None else loss


def _cfg_dict(cfg: "Hyperparameters") -> dict:
    return dict(vocab_size=cfg.vocab_size, block_size=cfg.block_size,
                n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                dropout=0.0, bias=cfg.bias, n_experts=cfg.n_experts,
                top_k=cfg.top_k, ffn_mult=cfg.ffn_mult,
                capacity_factor=cfg.capacity_factor)


def _cfg_from_dict(d: dict) -> "Hyperparameters":
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


def _save(model: MoEGPT, cfg: "Hyperparameters", out_ckpt: str) -> None:
    tmp = out_ckpt + ".tmp"
    torch.save({"model": model.state_dict(), "config": _cfg_dict(cfg)}, tmp)
    os.replace(tmp, out_ckpt)  # atomic: a timer-kill never leaves a half-written ckpt


class _MemmapLoader:
    """nanoGPT-style flat uint16 shard loader (frozen train shard bytes)."""

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

    model = MoEGPT(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] MoE {n_params/1e6:.1f}M total params | E={args.n_experts} top_k={args.top_k} "
          f"layers={args.n_layer} d={args.n_embd} | micro={args.micro_batch} accum={args.grad_accum} "
          f"seq={args.train_seq_len}", flush=True)

    loader = _MemmapLoader(args.data_path, args.train_seq_len, device, args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
                            fused=True)

    start = time.time()
    deadline = start + args.wallclock_sec * 0.94          # soft self-stop; external timer is the hard cap
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
                aux, z = model.aux_losses()
                loss = ce + args.aux_loss_coef * aux + args.router_z_coef * z
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        if step % 25 == 0:
            print(f"[train] step {step} lr {lr:.2e} ce {ce.item():.4f} aux {float(aux):.4f} "
                  f"peakGB {torch.cuda.max_memory_allocated()/1e9:.2f} t {time.time()-start:.0f}s", flush=True)
        if time.time() - last_save >= args.save_every_sec:
            _save(model, args, args.out_ckpt)
            last_save = time.time()
    _save(model, args, args.out_ckpt)
    print(f"[train] done: {step} steps in {time.time()-start:.0f}s -> {args.out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
