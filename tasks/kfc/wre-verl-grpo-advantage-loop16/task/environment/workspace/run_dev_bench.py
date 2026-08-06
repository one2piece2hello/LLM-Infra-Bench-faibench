"""Public dev smoke — solver-visible. Runs your submission on a small PUBLIC case and prints
shapes/values. It does NOT report correctness or performance (that is graded on hidden
workloads). Use it to sanity-check your implementation locally.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "submission")
import advantage_estimators as A  # noqa: E402


def main():
    torch.manual_seed(0)
    B, L, G = 64, 32, 8
    r = torch.randn(B, L)
    lengths = torch.randint(1, L + 1, (B,))
    mask = (torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)).float()
    index = np.arange(B) % G
    try:
        adv, ret = A.compute_grpo_outcome_advantage(r, mask, index)
        print("GRPO advantage shape:", tuple(adv.shape), "mean:", float(adv.mean()))
        adv2, _ = A.compute_rloo_outcome_advantage(r, mask, index)
        print("RLOO advantage shape:", tuple(adv2.shape), "mean:", float(adv2.mean()))
        print("dev smoke ran (this does NOT report correctness)")
    except NotImplementedError as e:
        print("not implemented yet:", e)


if __name__ == "__main__":
    main()
