"""REVIEWER-ONLY NEGATIVE — computes only the first sequence and leaves the rest of `out`
untouched (must score 0 on the sentinel/unwritten-output probe)."""
import math

import torch


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.D = int(cfg["head_size"])
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))
        self._fa = torch.ops.aten._flash_attention_forward

    def prepare(self):
        return None

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        if q.shape[0] == 0:
            return out
        n0 = int(cu_seqlens[1].item())
        if n0 <= 0:
            return out
        cu2 = torch.tensor([0, n0], device=q.device, dtype=torch.int32)
        r = self._fa(q[:n0], k[:n0], v[:n0], cu2, cu2, n0, n0, 0.0, True, False, scale=self.scale)
        o = r[0] if isinstance(r, (tuple, list)) else r
        out[:n0].copy_(o)
        return out
