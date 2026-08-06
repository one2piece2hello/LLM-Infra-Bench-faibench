#!/usr/bin/env python3
"""REVIEWER-ONLY strong reference recipe for e2e-a8-peft-adapter-byte-golf.

This is the 1.0 anchor of the open-ended reward (the 1.0 calibration anchor): a well-tuned,
int8-quantized LoRA whose rank budget is greedily allocated across the highest-leverage
projections and then SHRUNK until the packed artifact actually fits the byte cap
(measured, not estimated). It is deliberately stronger than the visible starter so that
config-only tweaks (bumping rank / adding a target / training longer) cannot beat it —
beating it needs real byte-efficiency work (sub-8-bit adapter quantization, shared /
seeded random bases à la VeRA, DoRA magnitude vectors, norm-and-bias tuning, adaptive
rank redistribution, tensor-train factorization, a leaner container).

NEVER model-visible: baked root-0700 at /opt/strong_baseline/ and run only by
VERIFIER_MODE=strong_baseline during authoring calibration.
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

BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "/app/base_model")
DATA_DIR = Path(os.environ.get("A8_DATA_DIR", "/data/datasets/a8_syscode"))
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))
CACHE = Path(os.environ.get("A8_TOKEN_CACHE", "/tmp/a8_tokens"))
ARTIFACT_BUDGET = int(os.environ.get("A8_ARTIFACT_BUDGET", 262_144))

# importance-ordered allocation waves (each wave = one rank unit on those projections)
WAVES = [("q_proj", "v_proj"), ("down_proj",), ("k_proj", "o_proj"), ("gate_proj",),
         ("up_proj",), ("q_proj", "v_proj"), ("down_proj",)]
ALPHA = 8.0
STEPS = 1400
SEQ_LEN = 1024
MICRO_BATCH = 2
ACCUM = 4
LR = 3e-4
WARMUP = 40
SEED = 1234


def load_tokens(tok, name: str) -> torch.Tensor:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{name}.pt"
    if cached.exists():
        return torch.load(cached, map_location="cpu")
    text = (DATA_DIR / f"{name}.txt").read_text(encoding="utf-8", errors="replace")
    ids: list[int] = []
    for i in range(0, len(text), 400_000):
        ids.extend(tok(text[i:i + 400_000], add_special_tokens=False)["input_ids"])
    t = torch.tensor(ids, dtype=torch.int64)
    torch.save(t, cached)
    return t


def batches(tokens, seq_len, bs, seed):
    g = torch.Generator().manual_seed(seed)
    n = (tokens.numel() - 1) // seq_len
    while True:
        for i in torch.randperm(n, generator=g).tolist():
            yield tokens[i * seq_len: i * seq_len + seq_len + 1]


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


def set_submodule(root, name, new):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() and hasattr(parent, "__getitem__") else getattr(parent, p)
    setattr(parent, parts[-1], new)


def plan_allocation(model, budget_bytes: int) -> dict[str, int]:
    """Greedy importance-ordered rank allocation under an int8 byte estimate."""
    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    alloc: dict[str, int] = {}
    used = 0
    for wave in WAVES:
        for name, mod in linears.items():
            if name.rsplit(".", 1)[-1] not in wave:
                continue
            cost = mod.in_features + mod.out_features       # 1 byte/param at int8
            if used + cost > budget_bytes:
                continue
            alloc[name] = alloc.get(name, 0) + 1
            used += cost
    return alloc


def pack(loras: dict[str, TrainLoRA], alpha: float, keep: set[str] | None = None) -> dict:
    chunks, index, off = [], {}, 0
    for name, m in loras.items():
        if keep is not None and name not in keep:
            continue
        pair = {}
        for key, t in (("A", m.A.detach()), ("B", m.B.detach())):
            flat = t.reshape(-1).float().cpu()
            amax = float(flat.abs().max().item()) or 1.0
            scale = amax / 127.0
            q = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8)
            pair[key] = {"off": off, "numel": q.numel(), "shape": list(t.shape), "scale": scale}
            chunks.append(q)
            off += q.numel()
        index[name] = pair
    blob = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int8)
    return {"format": "lora_pack_v1", "alpha": alpha, "compute_dtype": "bf16",
            "blob": blob, "index": index}


def save_fitting(loras, alpha, out: Path, budget: int) -> int:
    """Pack, MEASURE the real file size, and drop the lowest-priority modules until it
    fits (a measured cap, never an estimate). `loras` is in allocation-priority order,
    so popping the tail always drops the least important module first."""
    keep = list(loras.keys())
    while True:
        torch.save(pack(loras, alpha, set(keep)), out)
        size = out.stat().st_size
        print(f"[strong] packed {len(keep)} modules -> {size:,} bytes (budget {budget:,})", flush=True)
        if size <= budget or len(keep) <= 1:
            return size
        keep.pop()


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

    # leave ~10% of the cap for the container + index overhead; save_fitting measures for real
    alloc = plan_allocation(model, int(ARTIFACT_BUDGET * 0.90))
    loras: dict[str, TrainLoRA] = {}
    for name, r in alloc.items():
        mod = model.get_submodule(name)
        lora = TrainLoRA(mod, r, ALPHA, device)
        set_submodule(model, name, lora)
        loras[name] = lora
    params = [p for m in loras.values() for p in (m.A, m.B)]
    print(f"[strong] {len(loras)} adapted linears, {sum(p.numel() for p in params):,} trainable params",
          flush=True)

    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.5 * (1 + math.cos(math.pi * min(s / STEPS, 1.0)))))
    stream = batches(train_tokens, SEQ_LEN, MICRO_BATCH, SEED)
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            local = torch.stack([next(stream) for _ in range(MICRO_BATCH)]).to(device)
            x, y = local[:, :-1], local[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"[strong] step {step:4d}/{STEPS} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    art = SUBMISSION_DIR / "adapter.bin"
    size = save_fitting(loras, ALPHA, art, ARTIFACT_BUDGET)
    here = Path(__file__).resolve().parent
    if (here / "adapter_entry.py").exists() and here.resolve() != SUBMISSION_DIR.resolve():
        shutil.copy2(here / "adapter_entry.py", SUBMISSION_DIR / "adapter_entry.py")
    ce = eval_ce(model, val_tokens, device)
    entry_bytes = (SUBMISSION_DIR / "adapter_entry.py").stat().st_size
    print(json.dumps({"adapter_bytes": size, "entry_bytes": entry_bytes,
                      "total_bytes": size + entry_bytes, "budget": ARTIFACT_BUDGET,
                      "visible_val_ce": ce, "n_modules": len(loras),
                      "train_seconds": round(time.time() - t0, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
