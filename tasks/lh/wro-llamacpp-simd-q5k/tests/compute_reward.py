#!/usr/bin/env python3
"""Reward is emitted directly by tests/test.sh (reward = baseline_ms/candidate_ms, gated on the
deterministic checksum). This stub exists only to satisfy the loop16 packaging contract."""
if __name__ == "__main__":
    print("WRO_NOTE reward is computed inline by test.sh")
