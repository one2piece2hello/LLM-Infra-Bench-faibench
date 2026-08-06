# NEGATIVE known-bad (not baked into the image): fast-but-WRONG. Uses silu(g) in place of silu'(g) in the gate
# gradient (i.e. forgets the activation derivative). The forward y is correct, but dx is wrong along
# the gate path -> FAILS the correctness gate on dx. Reviewer-only. (Memory is irrelevant: it never
# reaches the peak_bytes stage because correctness gates first.)
import torch


def gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, chunk_size):
    xf = x.float(); wg = w_gate.float(); wu = w_up.float(); wd = w_down.float(); go = grad_out.float()
    g = xf @ wg
    u = xf @ wu
    sig = torch.sigmoid(g)
    silu = g * sig
    a = silu * u
    y = a @ wd
    da = go @ wd.t()
    du = da * silu
    dsilu = da * u
    # WRONG: uses silu(g) as the gate-path multiplier instead of silu'(g) = sig*(1 + g*(1 - sig)).
    dg = dsilu * silu
    dx = dg @ wg.t() + du @ wu.t()
    return y.to(torch.bfloat16), dx.to(torch.bfloat16)


def custom_kernel(data):
    x, w_gate, w_up, w_down, grad_out, config = data
    return gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, config["chunk_size"])
