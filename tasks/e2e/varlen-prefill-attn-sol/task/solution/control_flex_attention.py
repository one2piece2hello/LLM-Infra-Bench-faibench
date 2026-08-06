"""REVIEWER-ONLY CONTROL (invariant 1) — the strongest OFF-THE-SHELF alternative to the baseline:
torch.compile'd flex_attention with a document-causal BlockMask (GQA expanded up front).

If this beat the strong baseline on the scored suite, the baseline would not be hardened (design
§Principle 2 row 2) and would have to be replaced by a dispatch over both. Measured at authoring.
"""
import math

import torch


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.Hq = int(cfg["num_q_heads"]); self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"]); self.rep = self.Hq // self.Hkv
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))
        self._cache = {}
        self._fa = None

    def prepare(self):
        from torch.nn.attention.flex_attention import flex_attention
        self._fa = torch.compile(flex_attention, dynamic=False)

    def _mask(self, cu_seqlens, tot):
        from torch.nn.attention.flex_attention import create_block_mask
        key = (tot, int(cu_seqlens.numel()), int(cu_seqlens.sum().item()))
        if key in self._cache:
            return self._cache[key]
        cul = cu_seqlens.tolist()
        doc = torch.zeros(tot, device=cu_seqlens.device, dtype=torch.int32)
        for i in range(len(cul) - 1):
            doc[int(cul[i]):int(cul[i + 1])] = i

        def mask_mod(b, h, qi, ki):
            return (qi >= ki) & (doc[qi] == doc[ki])

        bm = create_block_mask(mask_mod, None, None, tot, tot, device=str(doc.device))
        self._cache[key] = bm
        return bm

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        tot = q.shape[0]
        if tot == 0:
            return out
        bm = self._mask(cu_seqlens, tot)
        qq = q.transpose(0, 1).unsqueeze(0)
        kk = k.transpose(0, 1).unsqueeze(0)
        vv = v.transpose(0, 1).unsqueeze(0)
        if self.rep > 1:
            kk = kk.repeat_interleave(self.rep, dim=1)
            vv = vv.repeat_interleave(self.rep, dim=1)
        o = self._fa(qq, kk, vv, block_mask=bm, scale=self.scale)
        out.copy_(o.squeeze(0).transpose(0, 1))
        return out
