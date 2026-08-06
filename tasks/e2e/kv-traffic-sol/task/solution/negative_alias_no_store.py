#!/usr/bin/env python3
"""NEGATIVE PROBE (reviewer-only) — "storage" by ALIASING the caller's buffers.

scatter() keeps a reference to k_src/v_src instead of writing the bytes into the pool, and
gather() replays those references. Must score 0: the harness overwrites the source buffers
after every scatter (the no-alias probe), so the replay returns poison.
"""
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _kb_base import BaseEngine


class KVTrafficEngine(BaseEngine):
    def allocate(self):
        super().allocate()
        self._alias = {}

    def scatter(self, layer, k_src, v_src):
        self._alias[layer] = (k_src, v_src)

    def gather(self, layer, k_out, v_out):
        a = self._alias.get(layer)
        if a is None or a[0].shape != k_out.shape:
            return super().gather(layer, k_out, v_out)
        k_out.copy_(a[0])
        v_out.copy_(a[1])
