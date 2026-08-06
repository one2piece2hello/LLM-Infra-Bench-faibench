"""REVIEWER-ONLY NEGATIVE — materialises the whole per-sequence score matrix in fp32 with no
row chunking (must score 0 on the workspace-allocation budget once a long sequence appears)."""
import math

import torch


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.Hq = int(cfg["num_q_heads"]); self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"]); self.rep = self.Hq // self.Hkv
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))

    def prepare(self):
        return None

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        if q.shape[0] == 0:
            return out
        cul = cu_seqlens.tolist()
        for i in range(len(cul) - 1):
            a, b = int(cul[i]), int(cul[i + 1])
            n = b - a
            if n <= 0:
                continue
            qq = q[a:b].float().transpose(0, 1)
            kk = k[a:b].float(); vv = v[a:b].float()
            if self.rep > 1:
                kk = kk.repeat_interleave(self.rep, dim=1); vv = vv.repeat_interleave(self.rep, dim=1)
            kk = kk.transpose(0, 1); vv = vv.transpose(0, 1)
            s = torch.matmul(qq, kk.transpose(1, 2)) * self.scale          # [Hq, n, n] fp32
            idx = torch.arange(n, device=q.device)
            mask = (idx.unsqueeze(0) > idx.unsqueeze(1)).unsqueeze(0).expand_as(s)
            s = s.masked_fill(mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            out[a:b] = torch.matmul(p, vv).transpose(0, 1).to(out.dtype)
        return out
