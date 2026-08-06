# Reviewer-only BASELINE2 — a FULL naive-but-correct implementation of the subsystem.
# NEVER baked into the solver image; scored through the reviewer patch slot.
# MUST pass correctness AND score 0 < vs_oracle < 1 (the correct-but-slow band).
# This is the idiomatic-naive form (verl's own per-group Python loop) — no strawman.
# Source-shape: volcengine/verl @ e7e052a naive estimators + a per-group Python reduction.
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def as_torch_index(index, device=None) -> torch.Tensor:
    target = torch.device(device) if device is not None else torch.device("cpu")
    if isinstance(index, torch.Tensor):
        t = index.reshape(-1)
        if t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            return t.to(device=target, dtype=torch.long)
        arr = np.array([str(x.item()) if hasattr(x, "item") else str(x) for x in t], dtype=object)
    else:
        arr = np.asarray(index).reshape(-1)
        if arr.dtype != object and np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr.astype(np.int64, copy=False)).to(device=target)
        if arr.dtype != object:
            arr = arr.astype(object)
    _, inv = np.unique(arr, return_inverse=True)
    return torch.from_numpy(inv.astype(np.int64, copy=False)).to(device=target)


@torch.no_grad()
def group_mean_std(scores, gidx, eps: float = 1e-6, device=None):
    # NAIVE: per-group Python loop with a defaultdict + per-group torch.stack/mean/std.
    target = torch.device(device) if device is not None else torch.device("cpu")
    scores = scores.reshape(-1).to(device=target, dtype=torch.float32)
    gidx = gidx.reshape(-1).to(device=target, dtype=torch.long)
    if scores.numel() != gidx.numel():
        raise ValueError(f"scores and gidx length mismatch: {scores.numel()} vs {gidx.numel()}")
    G = int(torch.max(gidx).item()) + 1 if gidx.numel() > 0 else 0
    if G == 0:
        empty = torch.empty(0, device=target, dtype=torch.float32)
        return empty, empty, empty
    id2score = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[int(gidx[i].item())].append(scores[i])
    mean = torch.zeros(G, device=target, dtype=torch.float32)
    std = torch.ones(G, device=target, dtype=torch.float32)
    count = torch.zeros(G, device=target, dtype=torch.float32)
    for idx, lst in id2score.items():
        count[idx] = float(len(lst))
        if len(lst) == 1:
            mean[idx] = 0.0
            std[idx] = 1.0
        else:
            st = torch.stack(lst)
            mean[idx] = torch.mean(st)
            v = torch.var(st, unbiased=True)
            std[idx] = torch.sqrt(torch.clamp(v, min=eps))
    return mean, std, count


@torch.no_grad()
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
                                   config=None):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    idx_list = [_hashable(index[i]) for i in range(bsz)]
    for i in range(bsz):
        id2score[idx_list[i]].append(scores[i])
    for k, lst in id2score.items():
        if len(lst) == 1:
            id2mean[k] = torch.tensor(0.0); id2std[k] = torch.tensor(1.0)
        else:
            st = torch.stack(lst); id2mean[k] = torch.mean(st); id2std[k] = torch.std(st)
    out = scores.clone()
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            out[i] = (scores[i] - id2mean[idx_list[i]]) / (id2std[idx_list[i]] + epsilon)
        else:
            out[i] = scores[i] - id2mean[idx_list[i]]
    out = out.unsqueeze(-1) * response_mask
    return out, out


@torch.no_grad()
def compute_rloo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    bsz = scores.shape[0]
    idx_list = [_hashable(index[i]) for i in range(bsz)]
    for i in range(bsz):
        id2score[idx_list[i]].append(scores[i])
    for k, lst in id2score.items():
        id2mean[k] = torch.tensor(0.0) if len(lst) == 1 else torch.mean(torch.stack(lst))
    out = scores.clone()
    for i in range(bsz):
        rn = len(id2score[idx_list[i]])
        if rn > 1:
            out[i] = scores[i] * rn / (rn - 1) - id2mean[idx_list[i]] * rn / (rn - 1)
    out = out.unsqueeze(-1) * response_mask
    return out, out


def _hashable(x):
    return x.item() if isinstance(x, torch.Tensor) else x
