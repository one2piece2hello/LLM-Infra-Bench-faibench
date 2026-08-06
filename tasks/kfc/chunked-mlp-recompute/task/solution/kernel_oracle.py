# Reviewer-only ORACLE (not baked into the image): chunked activation-recompute of the gated-MLP fwd+bwd.
# Never baked into any image. Used only to (a) calibrate the oracle peak_bytes constant in oracle mode and
# (b) prove the correctness + memory-headroom gradient. It processes the T rows in blocks of
# chunk_size and RECOMPUTES the [chunk, I] activations inside each block's backward instead of
# materializing the full [T, I] intermediates -> the peak allocator high-water holds only one
# block's activations -> low peak_bytes (the vs_oracle=1.0 anchor). Matmuls stay bf16 (H20 tensor
# cores accumulate in fp32); the elementwise activation math is done in fp32 on the small block.
import torch


def gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, chunk_size):
    T, H = x.shape
    I = w_gate.shape[1]
    y = torch.empty(T, H, device=x.device, dtype=torch.bfloat16)
    dx = torch.empty(T, H, device=x.device, dtype=torch.bfloat16)
    cs = int(chunk_size) if chunk_size and int(chunk_size) > 0 else T
    cs = max(1, min(cs, T))
    for lo in range(0, T, cs):                       # row-block loop; recompute per chunk
        hi = min(lo + cs, T)
        xc = x[lo:hi]                                # [c, H] bf16
        goc = grad_out[lo:hi]                        # [c, H] bf16
        # forward (recomputed) — only [c, I] intermediates are ever live
        gc = (xc @ w_gate).float()                   # [c, I]
        uc = (xc @ w_up).float()                     # [c, I]
        sig = torch.sigmoid(gc)
        siluc = gc * sig
        ac = siluc * uc                              # [c, I]
        y[lo:hi] = (ac.to(torch.bfloat16) @ w_down).to(torch.bfloat16)
        # backward for this block (activations recomputed above are reused here, then freed)
        dac = (goc @ w_down.t()).float()             # [c, I]
        duc = dac * siluc
        dsiluc = dac * uc
        silup = sig * (1.0 + gc * (1.0 - sig))       # silu'(g)
        dgc = dsiluc * silup
        dxc = dgc.to(torch.bfloat16) @ w_gate.t() + duc.to(torch.bfloat16) @ w_up.t()
        dx[lo:hi] = dxc.to(torch.bfloat16)
    return y, dx


def custom_kernel(data):
    x, w_gate, w_up, w_down, grad_out, config = data
    return gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, config["chunk_size"])
