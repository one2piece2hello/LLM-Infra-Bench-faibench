"""Hidden verifier assets — workload generation + correctness + timed benchmark.

Reviewer/verifier-only. Never baked into the solver workspace (root-0700 in the image or
uploaded fresh with tests/). Fully seeded so (shape, seed) -> bit-identical inputs (§A4).
"""
from __future__ import annotations

import numpy as np
import torch


def make_workload(spec: dict):
    """Deterministic workload. spec keys:
      B (int batch), L (int response length), G (int num groups), group_size_jitter (bool),
      seed (int), dist ('normal'|'sparse'), singleton_frac (float).
    Returns dict(token_level_rewards, response_mask, index) as torch tensors / np array.
    """
    g = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
    B, L, G = int(spec["B"]), int(spec["L"]), int(spec["G"])
    dist = spec.get("dist", "normal")

    # --- build group index of length B over G groups ---
    if spec.get("group_size_jitter", True):
        # uneven group sizes: assign each sample a group by a seeded multinomial-ish draw
        raw = torch.rand(B, generator=g)
        idx = (raw * G).long().clamp_(0, G - 1)
    else:
        idx = torch.arange(B) % G
    # inject some singletons deterministically
    sf = float(spec.get("singleton_frac", 0.0))
    if sf > 0:
        nsingle = max(1, int(sf * G))
        # force the last nsingle group-ids to appear exactly once by remapping
        base = G  # fresh ids beyond G-1 -> as_torch_index will factorize, still contiguous
        for k in range(nsingle):
            # find first sample currently in group k and move it to a brand-new singleton id
            pos = (idx == k).nonzero()
            if pos.numel() > 0:
                idx[pos[0, 0]] = base + k
    index = idx.numpy().copy()

    # --- rewards ---
    if dist == "sparse":
        # mostly zero, occasional +/-1 (outcome-reward flavour)
        r = torch.zeros(B, L)
        hit = torch.rand(B, L, generator=g) > 0.9
        signs = (torch.rand(B, L, generator=g) > 0.5).float() * 2 - 1
        r = r + hit.float() * signs
    else:
        r = torch.randn(B, L, generator=g)
    # response mask: variable-length (right-padded) — a seeded per-row length
    lengths = (torch.rand(B, generator=g) * (L - 1)).long() + 1
    mask = (torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)).float()

    return {"token_level_rewards": r, "response_mask": mask, "index": index}


# ---- the workload suite: public (dev) disjoint from hidden (scored) ----
PUBLIC_CASES = [
    {"name": "pub_small", "B": 64, "L": 32, "G": 8, "seed": 11, "dist": "normal"},
]

HIDDEN_CASES = [
    # correctness axes
    {"name": "normal", "B": 256, "L": 64, "G": 32, "seed": 101, "dist": "normal"},
    {"name": "boundary_singleton", "B": 128, "L": 48, "G": 40, "seed": 202, "dist": "normal", "singleton_frac": 0.5},
    {"name": "boundary_onegroup", "B": 96, "L": 16, "G": 1, "seed": 303, "dist": "normal", "group_size_jitter": False},
    {"name": "degenerate_sparse", "B": 128, "L": 64, "G": 16, "seed": 404, "dist": "sparse"},
    {"name": "novel_manygroup", "B": 512, "L": 128, "G": 128, "seed": 505, "dist": "normal"},
    # perf regime: large B with many groups so scatter-reduce beats the per-group python loop
    {"name": "perf_bigbatch", "B": 4096, "L": 256, "G": 512, "seed": 606, "dist": "normal"},
]

# metamorphic: permuting samples within the batch permutes outputs identically.
