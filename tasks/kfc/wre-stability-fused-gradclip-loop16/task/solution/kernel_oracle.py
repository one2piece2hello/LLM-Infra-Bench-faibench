# ORACLE variant (reviewer-only; never baked into the image). FAST: use the fused foreach vectorized ops.
# One torch._foreach_norm computes every tensor's norm in a single fused launch; stack + norm gives
# the global norm; one torch._foreach_mul_ scales all tensors in a single fused launch.
import torch


def clip_grads_by_global_norm(grads, max_norm):
    if len(grads) == 0:
        return grads, torch.zeros((), device="cuda", dtype=torch.float32)
    norms = torch._foreach_norm(grads, 2.0)                    # per-tensor L2, one fused launch
    total_norm = torch.norm(torch.stack([n for n in norms]), 2.0)
    clip_coef = max_norm / (total_norm + 1e-6)
    if float(clip_coef) < 1.0:
        torch._foreach_mul_(grads, clip_coef)                  # scale all, one fused launch
    return grads, total_norm.to(torch.float32)


def custom_kernel(data):
    grads, config = data
    return clip_grads_by_global_norm(grads, config["max_norm"])
