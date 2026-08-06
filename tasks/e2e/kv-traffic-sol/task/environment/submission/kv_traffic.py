#!/usr/bin/env python3
"""Paged KV-cache traffic engine — SUBMISSION SKELETON.

This is a CORRECT but completely unoptimised implementation. It is your starting point: it
passes every correctness requirement and it is very slow. Everything in this file is yours to
change or replace (and so is everything else in /app/repo).

THE CONTRACT (the verifier drives your engine through exactly these methods)
---------------------------------------------------------------------------
class KVTrafficEngine:

    __init__(cfg)   cfg is a dict with:
        num_layers, num_kv_heads, head_size, page_size, num_pages,
        max_batch, max_pages_per_request, dtype ("bfloat16"/"float16"), device ("cuda")

    allocate()      Allocate your paged KV pool. The pool must hold, per layer, num_pages
                    pages of page_size token slots, each slot holding one key vector and one
                    value vector of num_kv_heads*head_size elements of `dtype`.
                    The INTERNAL LAYOUT IS YOURS TO CHOOSE, but the total memory your engine
                    allocates is measured and must not exceed 1.10x the nominal pool size
                    (2 * num_layers * num_pages * page_size * num_kv_heads * head_size *
                    itemsize bytes).

    begin_step(plan)  Called once immediately before a run of per-layer scatter() or gather()
                    calls, all of which use the SAME plan. `plan` is a dict:
                      "block_table"      cuda int32 [batch, max_pages_per_request]; entry j is
                                         the PHYSICAL page holding logical token positions
                                         [j*page_size, (j+1)*page_size) of that request.
                                         Unused entries are -1.
                      "ctx_lens"         cuda int32 [batch]; first logical token position of
                                         this step's range for each request (NOT necessarily a
                                         multiple of page_size).
                      "new_lens"         cuda int32 [batch]; how many consecutive logical
                                         positions this step covers for each request (may be 0).
                      "block_table_cpu", "ctx_lens_cpu", "new_lens_cpu"
                                         host int32 mirrors of the same three tensors.
                      "total_tokens"     python int == sum(new_lens).
                      "batch"            python int.
                    begin_step() is inside the measured region. A plan is valid only until the
                    next begin_step() call: you may cache work derived from it for the layers
                    of that step, but you must not reuse a previous step's plan.

    scatter(layer, k_src, v_src)   k_src/v_src are [total_tokens, num_kv_heads, head_size]
                    tensors of `dtype`, packed request-major: rows
                    [cumsum(new_lens)[b], cumsum(new_lens)[b+1]) are request b's tokens in
                    increasing logical position. Store them into `layer`'s pool at the logical
                    positions the plan describes. You may not keep a reference to k_src/v_src:
                    they are overwritten after the call returns.

    gather(layer, k_out, v_out)    k_out/v_out are [total_tokens, num_kv_heads, head_size]
                    tensors of `dtype` OWNED BY THE VERIFIER. Fill them, in the same
                    request-major packing, with the key/value vectors stored at the logical
                    positions the plan describes. EVERY element must be written: the verifier
                    pre-fills them with a poison value and requires the result to be
                    bit-identical to what was stored.

    copy_pages(layer, src_pages, dst_pages)   cuda int32 [n] tensors of physical page ids.
                    For every i, page dst_pages[i] of `layer` must become an exact copy of page
                    src_pages[i] (all page_size slots, keys and values). The two sets are
                    disjoint.

    reset()         Drop all state; the pool may be reused afterwards.

WHAT IS MEASURED
----------------
Correctness runs first over hidden configurations; failing it scores 0. Then the verifier times
complete traffic steps (one begin_step + all layers) with CUDA events it records itself, over
hidden shapes, and every timed step's result is verified bit-exactly. Values are bf16/fp16 and
must round-trip EXACTLY - no lossy re-encoding.
"""
import torch


class KVTrafficEngine:
    def __init__(self, cfg):
        self.L = int(cfg["num_layers"])
        self.Hkv = int(cfg["num_kv_heads"])
        self.D = int(cfg["head_size"])
        self.PAGE = int(cfg["page_size"])
        self.npages = int(cfg["num_pages"])
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
        self.device = cfg["device"]
        self.k_pool = None
        self.v_pool = None
        self.plan = None

    def allocate(self):
        # one page-major tensor per layer for keys and for values
        self.k_pool = torch.zeros(self.L, self.npages, self.PAGE, self.Hkv, self.D,
                                  dtype=self.dtype, device=self.device)
        self.v_pool = torch.zeros(self.L, self.npages, self.PAGE, self.Hkv, self.D,
                                  dtype=self.dtype, device=self.device)

    def begin_step(self, plan):
        self.plan = plan

    def _ranges(self):
        """Walk the plan on the host: (dst_row, physical_page, page_offset, count) runs."""
        ctx = self.plan["ctx_lens_cpu"].tolist()
        new = self.plan["new_lens_cpu"].tolist()
        bt = self.plan["block_table_cpu"]
        row = 0
        for b in range(len(new)):
            p = ctx[b]
            left = new[b]
            while left > 0:
                page = p // self.PAGE
                off = p % self.PAGE
                n = min(self.PAGE - off, left)
                yield row, int(bt[b, page]), off, n
                row += n
                p += n
                left -= n

    def scatter(self, layer, k_src, v_src):
        for row, page, off, n in self._ranges():
            self.k_pool[layer, page, off:off + n] = k_src[row:row + n]
            self.v_pool[layer, page, off:off + n] = v_src[row:row + n]

    def gather(self, layer, k_out, v_out):
        for row, page, off, n in self._ranges():
            k_out[row:row + n] = self.k_pool[layer, page, off:off + n]
            v_out[row:row + n] = self.v_pool[layer, page, off:off + n]

    def copy_pages(self, layer, src_pages, dst_pages):
        src = src_pages.tolist()
        dst = dst_pages.tolist()
        for s, d in zip(src, dst):
            self.k_pool[layer, d] = self.k_pool[layer, s]
            self.v_pool[layer, d] = self.v_pool[layer, s]

    def reset(self):
        self.plan = None
        if self.k_pool is not None:
            self.k_pool.zero_()
            self.v_pool.zero_()
