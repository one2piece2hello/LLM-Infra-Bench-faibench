# NEGATIVE variant (reviewer-only; never baked into the image). FAST-but-WRONG: it clips each tensor by its OWN
# per-tensor norm (independent local clipping) instead of by the shared GLOBAL norm. The scaling
# factors differ per tensor, so the result violates the global-norm contract -> correctness FAILS.
import torch


def clip_grads_by_global_norm(grads, max_norm):
    if len(grads) == 0:
        return grads, torch.zeros((), device="cuda", dtype=torch.float32)
    norms = torch._foreach_norm(grads, 2.0)
    total_norm = torch.norm(torch.stack([n for n in norms]), 2.0)
    for i in range(len(grads)):
        local = norms[i]                                       # BUG: per-tensor local norm, not global
        coef = max_norm / (local + 1e-6)
        if float(coef) < 1.0:
            grads[i].mul_(coef)
    return grads, total_norm.to(torch.float32)


def custom_kernel(data):
    grads, config = data
    return clip_grads_by_global_norm(grads, config["max_norm"])
