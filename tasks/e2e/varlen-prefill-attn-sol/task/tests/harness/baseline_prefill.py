"""FROZEN — the STRONG BASELINE varlen/causal prefill attention (reviewer-owned).

This is the 1.0 anchor of the open-ended reward: the strongest varlen causal prefill kernel that
the baked environment already provides, driven through the same published entry contract as the
candidate.

WHY THIS IS THE HARDENED CHOICE (measured on NVIDIA H20 at authoring, geomean sol_fraction over the
12-workload authoring sweep — see the scope card `strong_baseline.why_config_flips_cannot_beat_it`):

    torch built-in FlashAttention-2 varlen (this baseline)      0.437   <- best off-the-shelf
    vLLM triton_unified_attention (paged, causal)               0.214
    F.scaled_dot_product_attention, per-sequence loop           0.239
    F.scaled_dot_product_attention, padded batch + is_causal    0.184
    torch.compile'd flex_attention, document-causal BlockMask   0.126
    vLLM prefix_prefill / triton_flash_attention                do not run in this image
    vllm.vllm_flash_attn.flash_attn_varlen_func                 CUDA PTX toolchain mismatch (dead)
    naive chunked torch (the shipped starting point)            0.029

There is no configuration knob on this path to leave untuned — the tile shapes are compiled into
the aten kernel — so the only way past this anchor is a genuinely better attention kernel.
NEVER shipped model-visible.
"""
import math

import torch


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.Hq = int(cfg["num_q_heads"])
        self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"])
        self.dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
        self.device = cfg["device"]
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))
        self._fa = torch.ops.aten._flash_attention_forward

    def prepare(self):
        return None

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        if q.shape[0] == 0:
            return out
        mx = int(max_seqlen)
        try:
            r = self._fa(q, k, v, cu_seqlens, cu_seqlens, mx, mx, 0.0, True, False,
                         scale=self.scale)
            o = r[0] if isinstance(r, (tuple, list)) else r
            out.copy_(o)
            return out
        except Exception:
            return self._fallback(q, k, v, cu_seqlens, out)

    # a correctness-only path for shapes the aten kernel refuses (never used in the timed region)
    def _fallback(self, q, k, v, cu_seqlens, out):
        cul = cu_seqlens.tolist()
        rep = self.Hq // self.Hkv
        for i in range(len(cul) - 1):
            a, b = int(cul[i]), int(cul[i + 1])
            n = b - a
            if n <= 0:
                continue
            kk = k[a:b].float()
            vv = v[a:b].float()
            if rep > 1:
                kk = kk.repeat_interleave(rep, dim=1)
                vv = vv.repeat_interleave(rep, dim=1)
            kk = kk.transpose(0, 1)
            vv = vv.transpose(0, 1)
            for c0 in range(0, n, 256):
                c1 = min(n, c0 + 256)
                qq = q[a + c0:a + c1].float().transpose(0, 1)
                s = torch.matmul(qq, kk.transpose(1, 2)) * self.scale
                rows = torch.arange(c0, c1, device=q.device).unsqueeze(1)
                cols = torch.arange(n, device=q.device).unsqueeze(0)
                s = s.masked_fill((cols > rows).unsqueeze(0), float("-inf"))
                p = torch.softmax(s, dim=-1)
                out[a + c0:a + c1] = torch.matmul(p, vv).transpose(0, 1).to(out.dtype)
        return out
