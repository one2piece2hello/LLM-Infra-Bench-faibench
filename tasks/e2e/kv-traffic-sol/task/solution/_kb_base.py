#!/usr/bin/env python3
"""Reviewer-only shared base for the negative probes (never shipped in the task image).

A plain, correct index-based paged-KV engine — the same shape as the strong baseline — so each
negative probe only has to express ONE defect.
"""
import torch

try:
    import vllm._custom_ops as _vco
except Exception:  # noqa: BLE001
    _vco = None


class BaseEngine:
    def __init__(self, cfg):
        self.L = int(cfg["num_layers"])
        self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"])
        self.PAGE = int(cfg["page_size"])
        self.npages = int(cfg["num_pages"])
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
        self.device = cfg["device"]
        self.elems = self.Hkv * self.D
        self.pool = None
        self._idx = None

    def allocate(self):
        self.pool = torch.zeros(self.L, 2, self.npages, self.PAGE, self.Hkv, self.D,
                                dtype=self.dtype, device=self.device)
        self._flat = self.pool.view(self.L, 2, self.npages * self.PAGE, self.elems)

    def begin_step(self, plan):
        bt, ctx, new = plan["block_table"], plan["ctx_lens"], plan["new_lens"]
        T, B = int(plan["total_tokens"]), int(plan["batch"])
        if T == 0:
            self._idx = torch.empty(0, dtype=torch.long, device=self.device)
            return
        cu = torch.zeros(B + 1, dtype=torch.int32, device=new.device)
        torch.cumsum(new, 0, out=cu[1:])
        req = torch.repeat_interleave(
            torch.arange(B, device=new.device, dtype=torch.int32), new).long()
        pos = (torch.arange(T, device=new.device, dtype=torch.int32) - cu[:-1][req]
               + ctx[req]).long()
        self._idx = bt[req, pos // self.PAGE].long() * self.PAGE + (pos % self.PAGE)

    def gather(self, layer, k_out, v_out):
        k_out.view(-1, self.elems).copy_(self._flat[layer, 0][self._idx])
        v_out.view(-1, self.elems).copy_(self._flat[layer, 1][self._idx])

    def scatter(self, layer, k_src, v_src):
        self._flat[layer, 0].index_copy_(0, self._idx, k_src.view(-1, self.elems))
        self._flat[layer, 1].index_copy_(0, self._idx, v_src.view(-1, self.elems))

    def copy_pages(self, layer, src_pages, dst_pages):
        if _vco is not None:
            mapping = torch.stack([src_pages.long(), dst_pages.long()], dim=1).contiguous()
            _vco.copy_blocks([self.pool[layer, 0]], [self.pool[layer, 1]], mapping)
            return
        s, d = src_pages.long(), dst_pages.long()
        self.pool[layer, 0].index_copy_(0, d, self.pool[layer, 0].index_select(0, s))
        self.pool[layer, 1].index_copy_(0, d, self.pool[layer, 1].index_select(0, s))

    def reset(self):
        self._idx = None
        if self.pool is not None:
            self.pool.zero_()
