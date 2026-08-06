# Performance Optimization Task — submission entry point.
#
# Implement `clip_grads_by_global_norm` to the contract in instruction.md, then make it as fast as
# possible. This is the ONLY file you edit. Leaving the NotImplementedError in place scores 0.
import torch  # noqa: F401  (available; use torch)


def clip_grads_by_global_norm(grads, max_norm):
    """Clip a list of gradient tensors by their GLOBAL L2 norm (training stability).

    Exploding gradients destabilize training; the standard remedy scales every gradient by one
    shared factor so the concatenation of all gradients has L2 norm at most ``max_norm``.

    Contract (all correct implementations agree within tolerance):
      * ``grads``: a list of ``N`` CUDA ``float32`` tensors (arbitrary shapes) — the gradients.
      * ``total_norm = sqrt(sum over all tensors of sum(g*g))`` (the global L2 norm across every
        element of every tensor).
      * ``clip_coef = max_norm / (total_norm + 1e-6)``. If ``clip_coef < 1.0``, multiply EVERY
        gradient tensor by ``clip_coef`` (in place is fine); otherwise leave them unchanged.
      * Return the tuple ``(grads, total_norm)`` where ``grads`` is the (now possibly-scaled) list
        and ``total_norm`` is a 0-dim ``float32`` CUDA tensor holding the ORIGINAL global norm
        (before clipping).

    Args:
        grads:    list[torch.Tensor] of N float32 CUDA gradient tensors.
        max_norm: float, the maximum allowed global L2 norm.

    Return:
        (grads, total_norm): the scaled list and the original global L2 norm (0-dim tensor).
    """
    raise NotImplementedError("implement clip_grads_by_global_norm to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls. data = (grads, config), config = {"max_norm": float}.
    Already wired — returns (grads, total_norm)."""
    grads, config = data
    return clip_grads_by_global_norm(grads, config["max_norm"])
