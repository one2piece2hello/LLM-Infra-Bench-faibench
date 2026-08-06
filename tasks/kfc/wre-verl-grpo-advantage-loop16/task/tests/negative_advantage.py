# Reviewer-only NEGATIVE known-bad: a semantic-change speed hack that skips the group
# normalization (returns the raw per-sample score broadcast) — FAST but WRONG -> correctness
# gate must fail -> reward 0. Proves the correctness gate has teeth.
from __future__ import annotations

import numpy as np
import torch


def as_torch_index(index, device=None) -> torch.Tensor:
    if isinstance(index, torch.Tensor):
        return index.reshape(-1).to(torch.long)
    return torch.as_tensor(np.asarray(index).reshape(-1), dtype=torch.long)


def group_mean_std(scores, gidx, eps: float = 1e-6, device=None):
    G = int(gidx.max().item()) + 1 if gidx.numel() else 0
    z = torch.zeros(G, dtype=torch.float32)
    return z, torch.ones(G, dtype=torch.float32), torch.ones(G, dtype=torch.float32)


def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
                                   config=None):
    # WRONG: skips group-relative normalization entirely
    scores = token_level_rewards.sum(dim=-1)
    out = scores.unsqueeze(-1) * response_mask
    return out, out


def compute_rloo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    scores = token_level_rewards.sum(dim=-1)
    out = scores.unsqueeze(-1) * response_mask
    return out, out
