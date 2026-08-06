"""REVIEWER-ONLY NEGATIVE — sliding-window truncation (must score 0 on the parity gate).

"Attend only the last 256 keys" is the classic delete-work-for-speed cheat: it turns the causal
triangle into a band, so the FLOPs collapse. The fp32 full-prefix reference kills it.
"""
import math

import torch

WINDOW = 256


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
            kk = k[a:b].float(); vv = v[a:b].float()
            if self.rep > 1:
                kk = kk.repeat_interleave(self.rep, dim=1); vv = vv.repeat_interleave(self.rep, dim=1)
            kk = kk.transpose(0, 1); vv = vv.transpose(0, 1)
            cols = torch.arange(n, device=q.device).unsqueeze(0)
            for c0 in range(0, n, 256):
                c1 = min(n, c0 + 256)
                qq = q[a + c0:a + c1].float().transpose(0, 1)
                s = torch.matmul(qq, kk.transpose(1, 2)) * self.scale
                rows = torch.arange(c0, c1, device=q.device).unsqueeze(1)
                bad = (cols > rows) | (cols < rows - WINDOW)
                s = s.masked_fill(bad.unsqueeze(0), float("-inf"))
                p = torch.softmax(s, dim=-1)
                out[a + c0:a + c1] = torch.matmul(p, vv).transpose(0, 1).to(out.dtype)
        return out
