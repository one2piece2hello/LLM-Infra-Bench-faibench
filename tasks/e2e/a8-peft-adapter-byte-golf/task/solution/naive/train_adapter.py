#!/usr/bin/env python3
"""NAIVE-but-honest starter: fine-tune a rank-1 bf16 LoRA on q_proj/v_proj.

This is a WORKING starting point, not a contract. It trains on the visible corpus,
writes `adapter.bin` + copies `adapter_entry.py`, and reports the artifact size and the
visible-val cross-entropy. It fits the byte budget with room to spare and it is
deliberately unambitious — matching the grader's strong reference recipe requires real
PEFT design work.

    PATH=/opt/kernelbench-venv/bin:$PATH python3 train_adapter.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- config
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "/app/base_model")
DATA_DIR = Path(os.environ.get("A8_DATA_DIR", "/data/datasets/a8_syscode"))
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))
CACHE = Path(os.environ.get("A8_TOKEN_CACHE", "/tmp/a8_tokens"))

TARGETS = ("q_proj",)              # which nn.Linear names to adapt
LAYERS = None                      # None = every decoder layer
RANK = 1
ALPHA = 2.0
STORE_DTYPE = "bf16"               # "bf16" | "int8"  (int8 buys 2x the parameters/byte)
STEPS = 120
SEQ_LEN = 1024
MICRO_BATCH = 2
ACCUM = 4
LR = 5e-5
WARMUP = 20
SEED = 1234


# ----------------------------------------------------------------------- data utils
def load_tokens(tok, name: str) -> torch.Tensor:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{name}.pt"
    if cached.exists():
        return torch.load(cached, map_location="cpu")
    text = (DATA_DIR / f"{name}.txt").read_text(encoding="utf-8", errors="replace")
    ids: list[int] = []
    step = 400_000
    for i in range(0, len(text), step):
        ids.extend(tok(text[i:i + step], add_special_tokens=False)["input_ids"])
    t = torch.tensor(ids, dtype=torch.int64)
    torch.save(t, cached)
    print(f"[data] {name}: {t.numel():,} tokens", flush=True)
    return t


def batches(tokens: torch.Tensor, seq_len: int, bs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = (tokens.numel() - 1) // seq_len
    while True:
        for i in torch.randperm(n, generator=g).tolist():
            yield tokens[i * seq_len: i * seq_len + seq_len + 1]
        seed += 1


# --------------------------------------------------------------------- adapter utils
class TrainLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, device):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=device, dtype=torch.float32))
        nn.init.normal_(self.A, std=1.0 / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=torch.float32))
        self.scale = alpha / max(r, 1)

    def forward(self, x):
        out = self.base(x)
        h = F.linear(x.to(self.A.dtype), self.A)
        return out + self.scale * F.linear(h, self.B).to(out.dtype)


def set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() and hasattr(parent, "__getitem__") else getattr(parent, p)
    setattr(parent, parts[-1], new)


def attach(model, targets, layers, rank, alpha, device) -> dict[str, TrainLoRA]:
    out: dict[str, TrainLoRA] = {}
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        if layers is not None:
            idx = [int(p) for p in name.split(".") if p.isdigit()]
            if not idx or idx[0] not in layers:
                continue
        lora = TrainLoRA(mod, rank, alpha, device)
        set_submodule(model, name, lora)
        out[name] = lora
    return out


def pack(loras: dict[str, TrainLoRA], store_dtype: str, alpha: float) -> dict:
    """One flat blob + an index (keeps torch.save container overhead negligible)."""
    chunks, index, off = [], {}, 0
    for name, m in loras.items():
        pair = {}
        for key, t in (("A", m.A.detach()), ("B", m.B.detach())):
            flat = t.reshape(-1).float().cpu()
            if store_dtype == "int8":
                amax = float(flat.abs().max().item()) or 1.0
                scale = amax / 127.0
                q = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8)
            else:
                scale = 1.0
                q = flat.to(torch.bfloat16)
            pair[key] = {"off": off, "numel": q.numel(), "shape": list(t.shape), "scale": scale}
            chunks.append(q)
            off += q.numel()
        index[name] = pair
    blob = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int8)
    return {"format": "lora_pack_v1", "alpha": alpha, "compute_dtype": "bf16",
            "blob": blob, "index": index}


@torch.inference_mode()
def eval_ce(model, tokens, device, seq_len=SEQ_LEN, max_seqs=48) -> float:
    model.eval()
    n = min((tokens.numel() - 1) // seq_len, max_seqs)
    tot, cnt = 0.0, 0
    for i in range(n):
        local = tokens[i * seq_len: i * seq_len + seq_len + 1].to(device)
        x, y = local[:-1].reshape(1, -1), local[1:].reshape(1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        V = logits.shape[-1]
        flat, tgt = logits.reshape(-1, V), y.reshape(-1)
        for s in range(0, flat.shape[0], 4096):
            tot += float(F.cross_entropy(flat[s:s + 4096].float(), tgt[s:s + 4096], reduction="sum"))
            cnt += int(tgt[s:s + 4096].numel())
    model.train()
    return tot / max(cnt, 1)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
    train_tokens = load_tokens(tok, "train")
    val_tokens = load_tokens(tok, "val")
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, dtype=torch.bfloat16)
    model.config.use_cache = False
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    loras = attach(model, TARGETS, LAYERS, RANK, ALPHA, device)
    params = [p for m in loras.values() for p in (m.A, m.B)]
    n_train = sum(p.numel() for p in params)
    print(f"[peft] adapted {len(loras)} linears, {n_train:,} trainable params", flush=True)

    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.5 * (1 + math.cos(math.pi * min(s / STEPS, 1.0)))))
    stream = batches(train_tokens, SEQ_LEN, MICRO_BATCH, SEED)
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            seqs = [next(stream) for _ in range(MICRO_BATCH)]
            local = torch.stack(seqs).to(device)
            x, y = local[:, :-1], local[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == STEPS - 1:
            print(f"[train] step {step:4d}/{STEPS} loss {loss.item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    art = SUBMISSION_DIR / "adapter.bin"
    torch.save(pack(loras, STORE_DTYPE, ALPHA), art)
    here = Path(__file__).resolve().parent
    if (here / "adapter_entry.py").exists() and here != SUBMISSION_DIR:
        shutil.copy2(here / "adapter_entry.py", SUBMISSION_DIR / "adapter_entry.py")
    ce = eval_ce(model, val_tokens, device)
    entry_bytes = (SUBMISSION_DIR / "adapter_entry.py").stat().st_size if (SUBMISSION_DIR / "adapter_entry.py").exists() else 0
    print(json.dumps({"adapter_bytes": art.stat().st_size, "entry_bytes": entry_bytes,
                      "total_bytes": art.stat().st_size + entry_bytes,
                      "visible_val_ce": ce, "trainable_params": n_train,
                      "train_seconds": round(time.time() - t0, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
