#!/usr/bin/env python3
"""NEGATIVE PROBE (reviewer-only) — caches the FIRST plan forever.

begin_step() ignores every plan after the first. Must score 0: the harness issues two plans
back to back (the current-plan probe) and changes the plan every timed block.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _kb_base import BaseEngine


class KVTrafficEngine(BaseEngine):
    def begin_step(self, plan):
        if getattr(self, "_frozen", None) is None:
            self._frozen = plan
        super().begin_step(self._frozen)
