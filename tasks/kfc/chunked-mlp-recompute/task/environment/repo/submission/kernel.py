# Performance Optimization Task — submission entry point.
#
# Implement `gated_mlp_fwd_bwd` below to the contract in instruction.md, then make
# it use as little PEAK GPU MEMORY as possible. This is the ONLY file you edit.
#
# The verifier drives custom_kernel on hidden bf16 workloads and grades numerical
# correctness first (rtol=atol=2e-2 vs a seeded fp32 reference, on BOTH returned
# tensors), then the peak GPU memory the call uses. A submission that leaves
# NotImplementedError in place scores 0.

import torch


def gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, chunk_size):
    """Manual forward + input-gradient backward of a gated (SwiGLU) MLP block.

    Let ``silu(z) = z * sigmoid(z)``. The block computes, over the stated dims:

        Forward
            g = x @ w_gate                 # [T, I]
            u = x @ w_up                   # [T, I]
            a = silu(g) * u                # [T, I]
            y = a @ w_down                 # [T, H]

        Backward  (given grad_out = dL/dy [T, H], produce dx = dL/dx [T, H])
            da    = grad_out @ w_down^T                      # [T, I]
            du    = da * silu(g)                             # [T, I]
            dsilu = da * u                                   # [T, I]
            silup = sigmoid(g) * (1 + g * (1 - sigmoid(g)))  # = silu'(g), [T, I]
            dg    = dsilu * silup                            # [T, I]
            dx    = dg @ w_gate^T + du @ w_up^T              # [T, H]

    Args:
        x:          bfloat16 tensor, shape [T, H]
        w_gate:     bfloat16 tensor, shape [H, I]
        w_up:       bfloat16 tensor, shape [H, I]
        w_down:     bfloat16 tensor, shape [I, H]
        grad_out:   bfloat16 tensor, shape [T, H]   (= dL/dy)
        chunk_size: int  — a suggested row-block granularity (see instruction.md).

    Returns:
        (y, dx), both bfloat16 [T, H]. The returned values must NOT depend on
        chunk_size (it only affects how the work is scheduled, never the math).
    """
    raise NotImplementedError("implement gated_mlp_fwd_bwd to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    Args:
        data = (x, w_gate, w_up, w_down, grad_out, config) where
            x, grad_out: bfloat16 [T, H]
            w_gate, w_up: bfloat16 [H, I]
            w_down:       bfloat16 [I, H]
            config:       {"T": int, "H": int, "I": int, "chunk_size": int}

    Returns:
        (y, dx) — both bfloat16 [T, H]; see gated_mlp_fwd_bwd.
    """
    x, w_gate, w_up, w_down, grad_out, config = data
    return gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, config["chunk_size"])
