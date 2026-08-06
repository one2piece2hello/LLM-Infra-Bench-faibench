#!/usr/bin/env python3
"""FROZEN — e2e-b1-kv-traffic-sol STRONG BASELINE (reviewer-owned, never model-visible).

The 1.0 anchor: the paged-KV data-movement path the baked serving engine ALREADY offers,
configured the way a competent engineer would configure it, with the per-op winner chosen by
MEASUREMENT on H20 at authoring time (probe log probe1.log / calib*.log):

  * pool layout  = the FlashAttention KV layout vLLM itself uses,
                   [num_pages, page_size, num_kv_heads, head_size] per layer per K/V,
                   so a token's heads are contiguous and a page is one contiguous run;
  * begin_step   = build the flat slot index ONCE per step with vectorised torch ops and reuse
                   it for every layer (exactly what vLLM's model runner does with slot_mapping /
                   query_start_loc: one index build per forward pass, 36 layers consume it);
  * gather       = torch advanced indexing on the flat token view — MEASURED faster than
                   torch.index_select(out=...) (0.122 vs 0.107 of peak) and than vLLM's own
                   ops.gather_cache (0.106, and it requires page-aligned starts);
  * scatter      = Tensor.index_copy_ on the flat token view — MEASURED faster than vLLM's own
                   CUDA ops.reshape_and_cache_flash on every probed shape (0.203-0.227 vs
                   0.107-0.195 of peak);
  * copy_pages   = vLLM's own CUDA ops.copy_blocks — MEASURED 2.6-3.4x faster than a torch
                   index_select/index_copy_ pair (0.205-0.280 vs 0.075-0.081 of peak).

Nothing here is degraded (design invariant 1): every path is the fastest one available without
writing a new kernel, and the index build is hoisted out of the per-layer loop. Beating it
requires real kernel work.
"""
import torch

try:
    import vllm._custom_ops as _vco
except Exception:  # noqa: BLE001
    _vco = None


class KVTrafficEngine:
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

    # --- one index build per step, reused by every layer (the production pattern) ---
    def begin_step(self, plan):
        bt = plan["block_table"]
        ctx = plan["ctx_lens"]
        new = plan["new_lens"]
        T = int(plan["total_tokens"])
        B = int(plan["batch"])
        if T == 0:
            self._idx = torch.empty(0, dtype=torch.long, device=self.device)
            return
        cu = torch.zeros(B + 1, dtype=torch.int32, device=new.device)
        torch.cumsum(new, 0, out=cu[1:])
        req = torch.repeat_interleave(
            torch.arange(B, device=new.device, dtype=torch.int32), new).long()
        pos = (torch.arange(T, device=new.device, dtype=torch.int32) - cu[:-1][req]
               + ctx[req]).long()
        self._idx = (bt[req, pos // self.PAGE].long() * self.PAGE + (pos % self.PAGE))

    def gather(self, layer, k_out, v_out):
        idx = self._idx
        k_out.view(-1, self.elems).copy_(self._flat[layer, 0][idx])
        v_out.view(-1, self.elems).copy_(self._flat[layer, 1][idx])

    def scatter(self, layer, k_src, v_src):
        idx = self._idx
        self._flat[layer, 0].index_copy_(0, idx, k_src.view(-1, self.elems))
        self._flat[layer, 1].index_copy_(0, idx, v_src.view(-1, self.elems))

    def copy_pages(self, layer, src_pages, dst_pages):
        if _vco is not None:
            mapping = torch.stack([src_pages.long(), dst_pages.long()], dim=1).contiguous()
            _vco.copy_blocks([self.pool[layer, 0]], [self.pool[layer, 1]], mapping)
            return
        s = src_pages.long()
        d = dst_pages.long()
        self.pool[layer, 0].index_copy_(0, d, self.pool[layer, 0].index_select(0, s))
        self.pool[layer, 1].index_copy_(0, d, self.pool[layer, 1].index_select(0, s))

    def reset(self):
        self._idx = None
        if self.pool is not None:
            self.pool.zero_()
