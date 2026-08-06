#!/usr/bin/env python3
"""NEGATIVE PROBE (reviewer-only) — keeps a SECOND, gather-friendly copy of the whole cache.

Must score 0: allocate() then reserves ~2x the nominal paged-pool size and the harness's
dual-measured pool-footprint budget (1.10x) rejects it.
"""
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _kb_base import BaseEngine


class KVTrafficEngine(BaseEngine):
    def allocate(self):
        super().allocate()
        self.shadow = torch.zeros_like(self.pool)
