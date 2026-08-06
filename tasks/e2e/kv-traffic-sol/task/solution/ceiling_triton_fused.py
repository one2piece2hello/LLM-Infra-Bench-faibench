#!/usr/bin/env python3
"""REVIEWER-ONLY CEILING PROBE for e2e-b1-kv-traffic-sol — NEVER shipped in the task image.

Demonstrates that the open-ended reward is really open-ended: one fused Triton kernel per
traffic op, computing the page/slot address inline from the block table so the whole step is a
single pass of coalesced page-sized runs (no materialised slot index, no temporaries, one
kernel launch per layer). This is the reviewer's ">1.0" ceiling for DoD item 4; a solver may
well do better (wider vector loads, TMA/cp.async, one launch for all layers, a layout that
makes the page runs longer).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_k(cache, out, block_table, ctx_lens, new_lens, cu_new, bt_stride, MP,
              PAGE: tl.constexpr, ELEMS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    p = tl.program_id(1)
    n = tl.load(new_lens + b)
    start = tl.load(ctx_lens + b)
    first = start // PAGE
    lo = (first + p) * PAGE
    if (lo < start + n) & (first + p < MP):
        beg = tl.maximum(lo, start)
        end = tl.minimum(lo + PAGE, start + n)
        phys = tl.load(block_table + b * bt_stride + first + p)
        src0 = (phys * PAGE + (beg - lo)) * ELEMS
        dst0 = (tl.load(cu_new + b) + (beg - start)) * ELEMS
        cnt = (end - beg) * ELEMS
        for o in range(0, PAGE * ELEMS, BLOCK):
            off = o + tl.arange(0, BLOCK)
            m = off < cnt
            tl.store(out + dst0 + off, tl.load(cache + src0 + off, mask=m), mask=m)


@triton.jit
def _scatter_k(cache, src, block_table, ctx_lens, new_lens, cu_new, bt_stride, MP,
               PAGE: tl.constexpr, ELEMS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    p = tl.program_id(1)
    n = tl.load(new_lens + b)
    start = tl.load(ctx_lens + b)
    first = start // PAGE
    lo = (first + p) * PAGE
    if (lo < start + n) & (first + p < MP):
        beg = tl.maximum(lo, start)
        end = tl.minimum(lo + PAGE, start + n)
        phys = tl.load(block_table + b * bt_stride + first + p)
        dst0 = (phys * PAGE + (beg - lo)) * ELEMS
        src0 = (tl.load(cu_new + b) + (beg - start)) * ELEMS
        cnt = (end - beg) * ELEMS
        for o in range(0, PAGE * ELEMS, BLOCK):
            off = o + tl.arange(0, BLOCK)
            m = off < cnt
            tl.store(cache + dst0 + off, tl.load(src + src0 + off, mask=m), mask=m)


@triton.jit
def _pagecopy(cache, src, dst, PER: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    s = tl.load(src + i) * PER
    d = tl.load(dst + i) * PER
    off = j * BLOCK + tl.arange(0, BLOCK)
    m = off < PER
    tl.store(cache + d + off, tl.load(cache + s + off, mask=m), mask=m)


def _pow2_le(x, cap):
    v = 1
    while v * 2 <= min(x, cap):
        v *= 2
    return max(v, 32)


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
        self.per_page = self.PAGE * self.elems
        self.BLOCK = _pow2_le(self.per_page, 4096)
        self.CBLOCK = _pow2_le(self.per_page, 4096)
        self.pool = None

    def allocate(self):
        self.pool = torch.zeros(self.L, 2, self.npages * self.per_page,
                                dtype=self.dtype, device=self.device)

    def begin_step(self, plan):
        self.plan = plan
        new = plan["new_lens"]
        B = int(plan["batch"])
        self.B = B
        self.cu = torch.zeros(B + 1, dtype=torch.int32, device=new.device)
        torch.cumsum(new, 0, out=self.cu[1:])
        self.bt = plan["block_table"]
        self.ctx = plan["ctx_lens"]
        self.new = new
        self.MP = self.bt.shape[1]
        maxnew = int(plan["new_lens_cpu"].max()) if B else 0
        self.gp = max(1, (maxnew + self.PAGE - 1) // self.PAGE + 1)
        self.total = int(plan["total_tokens"])

    def gather(self, layer, k_out, v_out):
        if self.total == 0:
            return
        grid = (self.B, self.gp)
        _gather_k[grid](self.pool[layer, 0], k_out, self.bt, self.ctx, self.new, self.cu,
                        self.MP, self.MP, PAGE=self.PAGE, ELEMS=self.elems, BLOCK=self.BLOCK,
                        num_warps=8)
        _gather_k[grid](self.pool[layer, 1], v_out, self.bt, self.ctx, self.new, self.cu,
                        self.MP, self.MP, PAGE=self.PAGE, ELEMS=self.elems, BLOCK=self.BLOCK,
                        num_warps=8)

    def scatter(self, layer, k_src, v_src):
        if self.total == 0:
            return
        grid = (self.B, self.gp)
        _scatter_k[grid](self.pool[layer, 0], k_src, self.bt, self.ctx, self.new, self.cu,
                         self.MP, self.MP, PAGE=self.PAGE, ELEMS=self.elems, BLOCK=self.BLOCK,
                         num_warps=8)
        _scatter_k[grid](self.pool[layer, 1], v_src, self.bt, self.ctx, self.new, self.cu,
                         self.MP, self.MP, PAGE=self.PAGE, ELEMS=self.elems, BLOCK=self.BLOCK,
                         num_warps=8)

    def copy_pages(self, layer, src_pages, dst_pages):
        n = src_pages.numel()
        if n == 0:
            return
        nb = (self.per_page + self.CBLOCK - 1) // self.CBLOCK
        _pagecopy[(n, nb)](self.pool[layer, 0], src_pages, dst_pages, PER=self.per_page,
                           BLOCK=self.CBLOCK, num_warps=8)
        _pagecopy[(n, nb)](self.pool[layer, 1], src_pages, dst_pages, PER=self.per_page,
                           BLOCK=self.CBLOCK, num_warps=8)

    def reset(self):
        if self.pool is not None:
            self.pool.zero_()
