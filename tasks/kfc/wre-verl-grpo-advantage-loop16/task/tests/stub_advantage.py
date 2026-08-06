"""Group-relative advantage estimation subsystem — SUBMISSION ENTRY.

Implement the three functions below to the contract stated in instruction.md, then make them
as fast as possible on the benchmark workloads. The starting state raises NotImplementedError
by design; a submission that leaves any function unimplemented scores 0 (correctness gate).

You may add private helpers to THIS file. Do not import from any package that provides these
estimators; the implementation must be your own within this file.
"""
from __future__ import annotations

import torch


def as_torch_index(index, device=None) -> torch.Tensor:
    """Canonicalize arbitrary per-sample group labels into a contiguous 1-D torch.long tensor
    with values in [0, G-1], preserving first-appearance grouping.

    Args:
        index: a 1-D sequence/np.ndarray/torch.Tensor of N group labels (ints, or arbitrary
            hashable labels). Integer labels are used directly (cast to long); non-integer
            labels are factorized to contiguous ids.
        device: optional target torch device (default: cpu).
    Returns:
        torch.LongTensor of shape (N,) on `device`.
    """
    raise NotImplementedError("implement as_torch_index")


def group_mean_std(scores, gidx, eps: float = 1e-6, device=None):
    """Per-group mean, std, and count.

    Args:
        scores: (N,) float tensor of per-sample scalar scores.
        gidx:   (N,) integer tensor of group ids in [0, G-1].
        eps:    variance floor applied inside the sqrt.
        device: optional target torch device (default: cpu).
    Returns:
        (mean_g, std_g, count_g), each shape (G,) float32 on `device`, where
        G = max(gidx)+1. std uses Bessel correction: denom = max(count-1, 1).
        A singleton group (count==1) returns mean=0.0 and std=1.0. Empty input -> three
        length-0 tensors.
    """
    raise NotImplementedError("implement group_mean_std")


def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
                                   config=None):
    """GRPO outcome advantage (group-relative, outcome-only).

    Args:
        token_level_rewards: (B, L) float tensor; the scalar per-sample reward is the sum
            over the last (length) dim.
        response_mask: (B, L) float tensor; the scalar advantage is broadcast over L and
            multiplied by this mask.
        index: length-B sequence of group labels (see as_torch_index).
        epsilon: added to the group std before dividing.
        norm_adv_by_std_in_grpo: if True, advantage = (score - group_mean) / (group_std + eps);
            if False, advantage = score - group_mean (Dr.GRPO).
        config: unused placeholder (accept and ignore).
    Returns:
        (advantages, returns), both (B, L) float tensors and equal to each other. Singleton
        groups use mean=0, std=1 (so a lone sample's normalized advantage is its own score).
    """
    raise NotImplementedError("implement compute_grpo_outcome_advantage")


def compute_rloo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    """RLOO (leave-one-out) outcome advantage.

    Args:
        token_level_rewards: (B, L) float tensor; per-sample score = sum over last dim.
        response_mask: (B, L) float tensor; scalar advantage broadcast over L then masked.
        index: length-B sequence of group labels (see as_torch_index).
        epsilon: unused for RLOO (accept for signature compatibility).
        config: unused placeholder.
    Returns:
        (advantages, returns), both (B, L) and equal. For a group of size n>1 the per-sample
        advantage is score_i * n/(n-1) - group_mean * n/(n-1) (leave-one-out baseline);
        a singleton group (n==1) yields its own raw score score_i (group_mean=0, factor=1), not 0.
    """
    raise NotImplementedError("implement compute_rloo_outcome_advantage")
