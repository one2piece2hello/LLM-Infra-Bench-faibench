"""REVIEWER-ONLY NEGATIVE — memoises the first result and replays it for every later call
(must score 0 on the stale-output probe: two different workloads cannot give one output)."""
import math

import torch


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.D = int(cfg["head_size"])
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))
        self._fa = torch.ops.aten._flash_attention_forward
        self._cache = None

    def prepare(self):
        self._cache = None

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        if q.shape[0] == 0:
            return out
        if self._cache is not None and self._cache.shape == out.shape:
            out.copy_(self._cache)
            return out
        mx = int(max_seqlen)
        r = self._fa(q, k, v, cu_seqlens, cu_seqlens, mx, mx, 0.0, True, False, scale=self.scale)
        o = r[0] if isinstance(r, (tuple, list)) else r
        out.copy_(o)
        self._cache = out.clone()
        return out
