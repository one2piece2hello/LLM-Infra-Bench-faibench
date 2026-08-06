# Reviewer-only ORACLE — the real vectorized verl implementation. NEVER baked into the
# solver-visible image. Uploaded fresh with tests/ at scoring time (root-0700).
# Source: volcengine/verl @ e7e052aba3af98115e4340c0228d404313fd8002
#   verl/utils/groupwise.py::group_mean_std (index_add_ scatter-reduce)
#   verl/trainer/ppo/core_algos.py::compute_grpo_vectorized_outcome_advantage
#   verl/trainer/ppo/core_algos.py::compute_rloo_vectorized_outcome_advantage
# This is BOTH the correctness golden source AND the perf 1.0 reference (vs_oracle anchor).
from __future__ import annotations

import numpy as np
import torch


def as_torch_index(index, device=None) -> torch.Tensor:
    """Canonicalize arbitrary group labels to a contiguous 1-D long tensor in [0..G-1]."""
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
    """Per-group mean/std/count in pure PyTorch, vectorized via index_add_ scatter-reduce.
    std uses Bessel correction (denom = max(count-1, 1)); singleton groups -> mean=0, std=1."""
    target = torch.device(device) if device is not None else torch.device("cpu")
    scores = scores.reshape(-1).to(device=target, dtype=torch.float32)
    gidx = gidx.reshape(-1).to(device=target, dtype=torch.long)
    if scores.numel() != gidx.numel():
        raise ValueError(f"scores and gidx length mismatch: {scores.numel()} vs {gidx.numel()}")
    G = int(torch.max(gidx).item()) + 1 if gidx.numel() > 0 else 0
    if G == 0:
        empty = torch.empty(0, device=target, dtype=torch.float32)
        return empty, empty, empty
    ones = torch.ones_like(scores, dtype=torch.float32)
    count = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, ones)
    s1 = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, scores)
    mean = s1 / count.clamp_min(1.0)
    centered = scores - mean[gidx]
    var_num = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, centered * centered)
    denom = (count - 1.0).clamp_min(1.0)
    var = var_num / denom
    std = torch.sqrt(torch.clamp(var, min=eps))
    single = count <= 1.0
    if torch.any(single):
        mean = mean.clone(); std = std.clone()
        mean[single] = 0.0; std[single] = 1.0
    return mean, std, count


@torch.no_grad()
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
                                   config=None):
    scores = token_level_rewards.sum(dim=-1)
    g = as_torch_index(index, device=scores.device)
    mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)
    if norm_adv_by_std_in_grpo:
        scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
    else:
        scalars = scores - mean_g[g]
    advantages = scalars.unsqueeze(-1) * response_mask
    return advantages, advantages


@torch.no_grad()
def compute_rloo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    scores = token_level_rewards.sum(dim=-1)
    g = as_torch_index(index, device=scores.device)
    # mean_g singleton-fallback = 0.0 (group_mean_std), matching naive RLOO's id2mean=0 for
    # singleton groups. factor = n/(n-1) for n>1, 1.0 for singleton -> (s - 0)*1 = raw s,
    # exactly reproducing the naive per-element RLOO leave-one-out baseline.
    mean_g, _, count_g = group_mean_std(scores, g, eps=0.0, device=scores.device)
    n = count_g[g]
    factor = torch.where(n > 1, n / (n - 1.0).clamp_min(1.0), torch.ones_like(n))
    adv = (scores - mean_g[g]) * factor
    adv = adv.unsqueeze(-1) * response_mask
    return adv, adv
