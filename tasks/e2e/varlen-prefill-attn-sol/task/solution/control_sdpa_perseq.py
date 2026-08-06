"""REVIEWER-ONLY CONTROL (invariant 1) — the obvious off-the-shelf alternative:
F.scaled_dot_product_attention called once per sequence with is_causal=True."""
import math

import torch
import torch.nn.functional as F


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.Hq = int(cfg["num_q_heads"]); self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"]); self.rep = self.Hq // self.Hkv
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))
        self.gqa = True

    def prepare(self):
        x = torch.zeros(1, self.Hq, 8, self.D, device="cuda", dtype=torch.bfloat16)
        y = torch.zeros(1, self.Hkv, 8, self.D, device="cuda", dtype=torch.bfloat16)
        try:
            F.scaled_dot_product_attention(x, y, y, is_causal=True, enable_gqa=(self.rep > 1))
            self.gqa = True
        except TypeError:
            self.gqa = False

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        if q.shape[0] == 0:
            return out
        cul = cu_seqlens.tolist()
        for i in range(len(cul) - 1):
            a, b = int(cul[i]), int(cul[i + 1])
            if b - a <= 0:
                continue
            qi = q[a:b].transpose(0, 1).unsqueeze(0)
            ki = k[a:b].transpose(0, 1).unsqueeze(0)
            vi = v[a:b].transpose(0, 1).unsqueeze(0)
            if self.gqa:
                o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=self.scale,
                                                   enable_gqa=(self.rep > 1))
            else:
                o = F.scaled_dot_product_attention(qi, ki.repeat_interleave(self.rep, 1),
                                                   vi.repeat_interleave(self.rep, 1),
                                                   is_causal=True, scale=self.scale)
            out[a:b] = o.squeeze(0).transpose(0, 1).to(out.dtype)
        return out
