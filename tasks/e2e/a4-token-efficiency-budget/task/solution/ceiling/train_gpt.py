# e2e-a4-token-efficiency-budget : the CEILING oracle (REVIEWER-ONLY -- never in the image).
#
# The public record trajectory at a FIXED TOKEN budget: Muon (spectral/orthogonalised updates
# via Newton-Schulz) on the hidden weight matrices + AdamW on embeddings/head, plus the
# modernized block (RMSNorm, RoPE, QK-norm, ReLU^2 MLP, zero-init residual projections,
# zero-init head, logit softcap).
#
# MEASURED on NVIDIA H20 at the 12M-token probe budget, identical frozen data stream:
#   naive AdamW      7.8369 bits/token -> reward 0.9887
#   tuned AdamW      7.7482  (the 1.0 anchor; best of a micro-batch x LR sweep)
#   THIS recipe      6.1447 bits/token -> reward 1.2610   (m32; m16 1.2584, m8 1.2488)
# Reference: github.com/KellerJordan/modded-nanogpt + kellerjordan.github.io/posts/muon/
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

VOCAB = 8192
SEQ = 512      # default block size referenced by ModGPT's signature


def rope(x, cos, sin):
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ModBlock(nn.Module):
    """modernized block: RMSNorm + RoPE + QK-norm + ReLU^2 MLP + zero-init projections."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        nn.init.zeros_(self.proj.weight)   # muP-like zero-init of residual projections
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        xn = F.rms_norm(x, (D,))
        q, k, v = self.qkv(xn).split(D, dim=2)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK-norm
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).contiguous().view(B, T, D))
        h = self.fc(F.rms_norm(x, (D,)))
        x = x + self.fc2(F.relu(h).square())      # ReLU^2
        return x


class ModGPT(nn.Module):
    def __init__(self, d=512, n=8, h=8, vocab=VOCAB, seq=SEQ):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([ModBlock(d, h) for _ in range(n)])
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.zeros_(self.head.weight)          # zero-init head
        hd = d // h
        t = torch.arange(seq)
        f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = t[:, None].float() * f[None]
        self.register_buffer("cos", ang.cos()[None, None], persistent=False)
        self.register_buffer("sin", ang.sin()[None, None], persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = F.rms_norm(self.wte(idx), (self.wte.weight.size(1),))
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(F.rms_norm(x, (x.size(-1),)))
        logits = 30.0 * torch.tanh(logits / 30.0)     # logit softcap (record trick)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))



def ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon for hidden 2D weights (KellerJordan/Muon, single-device form)."""

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                     ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                m.lerp_(p.grad, 1 - g["momentum"])
                upd = p.grad.lerp(m, g["momentum"])
                upd = ns5(upd, g.get("ns_steps", 5))
                upd = upd * max(1.0, upd.size(-2) / upd.size(-1)) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.reshape(p.shape).to(p.dtype), alpha=-g["lr"])




class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "/data/datasets/wikitext_bpe8k")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tokenizers/wikitext_bpe8k.model")
    seed = int(os.environ.get("SEED", "1337"))
    vocab_size = VOCAB
    train_seq_len = 512
    block_size = 512
    val_batch_size = 32_768
    n_layer, n_head, n_embd = 8, 8, 512
    micro_batch, grad_accum = 32, 1        # the best-measured record point
    learning_rate = 1.5e-3                 # AdamW group (embeddings + head)
    muon_lr = 0.04                         # Muon group (hidden 2-D weights)
    weight_decay = 0.1
    grad_clip = 1.0
    warmup_frac, cooldown_frac, min_lr_frac = 0.02, 0.40, 0.02
    save_every_sec = 20.0


class TokenStream:
    def __init__(self, data_path, seq_len, device):
        self.data = np.memmap(Path(data_path) / "train.bin", dtype=np.uint16, mode="r")
        self.seq_len, self.device, self.pos, self.tokens_served = seq_len, device, 0, 0

    def batch(self, n):
        need = n * self.seq_len + 1
        if self.pos + need >= len(self.data):
            self.pos = 0
        blk = np.asarray(self.data[self.pos:self.pos + need], dtype=np.int64)
        self.pos += n * self.seq_len
        self.tokens_served += n * self.seq_len
        x = torch.from_numpy(blk[:-1].reshape(n, self.seq_len))
        y = torch.from_numpy(blk[1:].reshape(n, self.seq_len))
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))


def _save(model, args, out_ckpt):
    m = getattr(model, "_orig_mod", model)
    tmp = str(out_ckpt) + ".tmp"
    torch.save({"model": m.state_dict(),
                "config": dict(d=args.n_embd, n=args.n_layer, h=args.n_head,
                               vocab=args.vocab_size, seq=args.block_size)}, tmp)
    os.replace(tmp, out_ckpt)


def load_model_for_verification(checkpoint_path, device):
    args = Hyperparameters()
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    c = obj.get("config", {})
    model = ModGPT(d=c.get("d", args.n_embd), n=c.get("n", args.n_layer),
                   h=c.get("h", args.n_head), vocab=c.get("vocab", args.vocab_size),
                   seq=c.get("seq", args.block_size))
    model.load_state_dict(obj["model"], strict=True)
    args.train_seq_len = c.get("seq", args.block_size)
    args.vocab_size = c.get("vocab", args.vocab_size)
    return args, model.to(device).eval()


def train_for_budget(seed, out_dir, budget_seconds):
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

    model = ModGPT(args.n_embd, args.n_layer, args.n_head, args.vocab_size, args.block_size).to(device)
    stream = TokenStream(args.data_path, args.train_seq_len, device)
    budget_tokens = int(stream.data.shape[0])
    tps = args.micro_batch * args.grad_accum * args.train_seq_len
    total_steps = max(1, budget_tokens // tps)

    hidden = [p for nm, p in model.named_parameters()
              if p.ndim == 2 and "wte" not in nm and "head" not in nm]
    other = [p for nm, p in model.named_parameters()
             if not (p.ndim == 2 and "wte" not in nm and "head" not in nm)]
    opts = [SingleDeviceMuon(hidden, lr=args.muon_lr, momentum=0.95, weight_decay=args.weight_decay),
            torch.optim.AdamW(other, lr=args.learning_rate, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay, fused=True)]
    print(f"[ceiling] seed={seed} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"budget={budget_tokens/1e6:.2f}M tokens -> {total_steps} steps", flush=True)

    warm = max(1, int(args.warmup_frac * total_steps))
    cool = max(1, int(args.cooldown_frac * total_steps))
    t0 = last_save = time.time()
    guard = float(budget_seconds) * 0.94 if budget_seconds else float("inf")
    model.train()
    for step in range(total_steps):
        if time.time() - t0 > guard:
            break
        if step < warm:
            f = (step + 1) / warm
        elif step > total_steps - cool:
            f = max(args.min_lr_frac, (total_steps - step) / cool)
        else:
            f = 1.0
        for o in opts:
            base = args.muon_lr if isinstance(o, SingleDeviceMuon) else args.learning_rate
            for g in o.param_groups:
                g["lr"] = base * f
            o.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = stream.batch(args.micro_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in opts:
            o.step()
        if step % 200 == 0:
            print(f"[ceiling] step {step}/{total_steps} loss {float(loss):.4f} "
                  f"tok {stream.tokens_served/1e6:.1f}M t {time.time()-t0:.0f}s", flush=True)
        if time.time() - last_save >= args.save_every_sec:
            _save(model, args, out_ckpt)
            last_save = time.time()
    _save(model, args, out_ckpt)
    print(f"[ceiling] done {stream.tokens_served/1e6:.2f}M tokens in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    train_for_budget(int(os.environ.get("SEED", "1337")),
                     os.environ.get("OUT_DIR", "/app/submission"),
                     float(os.environ.get("MAX_WALLCLOCK_SEC", "900")))
