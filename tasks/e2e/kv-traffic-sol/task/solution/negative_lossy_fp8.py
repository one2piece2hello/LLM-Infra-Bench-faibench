#!/usr/bin/env python3
"""NEGATIVE PROBE (reviewer-only) — LOSSY KV storage (values rounded through fp8).

The declared dtype is kept (so nothing crashes) but every stored value is round-tripped
through float8_e4m3fn first, i.e. the cache no longer holds what was written. Must score 0 on
the bit-exactness gate. This is the "compress the KV cache to move fewer bytes" cheat.
"""
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _kb_base import BaseEngine


class KVTrafficEngine(BaseEngine):
    def scatter(self, layer, k_src, v_src):
        f8 = torch.float8_e4m3fn
        super().scatter(layer, k_src.to(f8).to(k_src.dtype), v_src.to(f8).to(v_src.dtype))
