"""Varlen / causal PREFILL attention — THIS FILE IS YOURS TO REWRITE.

The verifier imports this file and drives the class `VarlenPrefillAttention` through the entry
contract documented in /app/instruction.md.  The implementation below is CORRECT but naive: for
each sequence in turn it gathers that sequence's keys and values, forms the score matrix in
float32 in row chunks, masks it, softmaxes it and multiplies by V.  It passes the correctness gate
and it is slow.

Replace as much of it as you like — the tiling, the masking strategy, the numerics, the memory
traffic, a Triton kernel, a CUDA extension you compile yourself (nvcc is available), anything.

Contract summary (see /app/instruction.md for the authoritative version):
    VarlenPrefillAttention(cfg)     cfg keys: num_q_heads, num_kv_heads, head_size, dtype,
                                    device, max_num_seqs, max_seq_len, max_total_tokens,
                                    causal (always True), softmax_scale
    .prepare()                      allocate whatever persistent workspace you need
    .forward(q, k, v, cu_seqlens, max_seqlen, out) -> out
                                    q: [total_tokens, num_q_heads,  head_size]
                                    k: [total_tokens, num_kv_heads, head_size]
                                    v: [total_tokens, num_kv_heads, head_size]
                                    cu_seqlens: int32 [num_seqs + 1], non-decreasing, cu[0] == 0,
                                                cu[-1] == total_tokens
                                    out: [total_tokens, num_q_heads, head_size], PRE-ALLOCATED —
                                         fill it and return it.
"""
import math

import torch

ROW_CHUNK = 256          # query rows processed per score-matrix chunk


class VarlenPrefillAttention:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.Hq = int(cfg["num_q_heads"])
        self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"])
        self.rep = self.Hq // self.Hkv
        self.dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
        self.device = cfg["device"]
        self.scale = float(cfg.get("softmax_scale") or 1.0 / math.sqrt(self.D))

    def prepare(self):
        """Nothing to pre-allocate in this naive version."""
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
            kk = k[a:b].float()
            vv = v[a:b].float()
            if self.rep > 1:
                kk = kk.repeat_interleave(self.rep, dim=1)
                vv = vv.repeat_interleave(self.rep, dim=1)
            kk = kk.transpose(0, 1)                                  # [Hq, n, D]
            vv = vv.transpose(0, 1)                                  # [Hq, n, D]
            cols = torch.arange(n, device=q.device).unsqueeze(0)
            for c0 in range(0, n, ROW_CHUNK):
                c1 = min(n, c0 + ROW_CHUNK)
                qq = q[a + c0:a + c1].float().transpose(0, 1)         # [Hq, c, D]
                s = torch.matmul(qq, kk.transpose(1, 2)) * self.scale  # [Hq, c, n]
                rows = torch.arange(c0, c1, device=q.device).unsqueeze(1)
                s = s.masked_fill((cols > rows).unsqueeze(0), float("-inf"))
                p = torch.softmax(s, dim=-1)
                out[a + c0:a + c1] = torch.matmul(p, vv).transpose(0, 1).to(out.dtype)
        return out
