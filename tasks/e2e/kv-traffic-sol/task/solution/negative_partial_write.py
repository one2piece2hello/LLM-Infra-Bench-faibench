#!/usr/bin/env python3
"""NEGATIVE PROBE (reviewer-only) — skips the tail of every gather.

Writes only the first 90% of the requested rows. Must score 0: the harness poisons the
verifier-owned output buffers before every gather, so unwritten rows are detected exactly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _kb_base import BaseEngine


class KVTrafficEngine(BaseEngine):
    def gather(self, layer, k_out, v_out):
        n = k_out.shape[0]
        if n == 0:
            return
        m = max(1, int(0.9 * n))
        idx = self._idx[:m]
        k_out.view(-1, self.elems)[:m].copy_(self._flat[layer, 0][idx])
        v_out.view(-1, self.elems)[:m].copy_(self._flat[layer, 1][idx])
