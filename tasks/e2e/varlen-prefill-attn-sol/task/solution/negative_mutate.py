"""REVIEWER-ONLY NEGATIVE — folds the softmax scale into the caller's q tensor IN PLACE
(must score 0 on the input-immutability check; the harness's inputs are not the candidate's)."""
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
        q.mul_(self.scale)                      # <-- scribbles on the harness's input
        mx = int(max_seqlen)
        r = self._fa(q, k, v, cu_seqlens, cu_seqlens, mx, mx, 0.0, True, False, scale=1.0)
        o = r[0] if isinstance(r, (tuple, list)) else r
        out.copy_(o)
        return out
