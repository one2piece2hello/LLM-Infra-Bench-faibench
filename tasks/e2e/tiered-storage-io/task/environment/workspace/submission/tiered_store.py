#!/usr/bin/env python3
"""Tiered storage engine -- IMPLEMENT ME.

This is the ONLY file you may edit. The verifier imports this module and exercises a single class you
must implement: `TieredStore`. See instruction.md for the full contract. Below is a runnable-but-
INCOMPLETE stub so the harness imports; it FAILS the graded case set until you implement size-bounded LRU
eviction, resumable segmented transfer with integrity checks, coalesced cold-miss fetch, and
cross-tier read consistency under concurrency.

Scoring is BINARY: reward = 1.0 only if EVERY graded case passes and no cheat / forbidden-path
gate trips; ANY failing case scores 0.0. There is no partial credit for a higher pass rate, so
"most of it works" is worth exactly the same as an empty file.

Required API (exact names/signatures):

  class TieredStore:
      def __init__(self, hot_capacity, cold_fetch_fn=None)
          # hot_capacity: max total bytes of VALUES resident in the hot tier.
          # cold_fetch_fn (optional): callable(key)->bytes used to fetch a cold-only object; when
          # given, a hot miss for a key that is only in the cold tier must call it (coalesced).
      def put(self, key, value)          # value: bytes. Durably record into cold; insert into hot
                                         # (most-recently-used); evict LRU keys until hot size <= cap.
      def get(self, key)                 # -> bytes | None. Hot hit -> touch & return. Hot miss but in
                                         # cold -> fetch into hot (touch, may evict) & return. Missing
                                         # everywhere -> None. Concurrent cold-miss fetches for the SAME
                                         # key must COALESCE (cold_fetch_fn called as few times as
                                         # possible) and all callers observe the identical value.
      def seed_cold(self, key, value)    # register a value in the REMOTE backing (reachable only via
                                         # cold_fetch_fn); NOT hot, NOT a local cold cache copy.
      def hot_size(self)                 # -> int total bytes of values currently resident in hot.
      def in_hot(self, key)              # -> bool whether key currently resides in the hot tier.
      def transfer(self, key, segment_size, sink=None, resume_from=None)
          # Fetch `key`'s object from the cold tier in fixed-size segments, writing each to `sink`
          # via sink.write_segment(index, data, crc). segment_size MUST be > 0 (else raise). A missing
          # key must raise (or return None). If resume_from (a manifest list of already-received
          # segment indices) is given, you must SKIP those indices (do not re-send). If write_segment
          # raises a bad-segment signal (crc/corruption), you MUST re-fetch and re-send that segment
          # (never skip it). Return the total number of segments in the object.
"""
import threading
import zlib


class TieredStore:
    def __init__(self, hot_capacity, cold_fetch_fn=None):
        self.hot_capacity = hot_capacity
        self._cold_fetch_fn = cold_fetch_fn
        self._hot = {}    # key -> value (NO ordering / eviction yet -- add it)
        self._cold = {}   # key -> value (authoritative, locally cached)
        self._remote = set()  # keys resident only in remote backing (fetch via cold_fetch_fn)
        self._lock = threading.Lock()

    # ---- STARTER STUB: replace with a correct implementation ----
    def put(self, key, value):
        self._cold[key] = value
        self._hot[key] = value
        # NOTE: no eviction, no LRU ordering, no capacity enforcement -- add it.

    def get(self, key):
        if key in self._hot:
            return self._hot[key]
        if key in self._cold:
            return self._cold[key]
        # NOTE: remote fetch via cold_fetch_fn + coalescing NOT implemented -- add it.
        return None

    def seed_cold(self, key, value):
        # stub: does not distinguish remote-only from local cold cache -- fix per contract.
        self._cold[key] = value

    def hot_size(self):
        return sum(len(v) for v in self._hot.values())

    def in_hot(self, key):
        return key in self._hot

    def transfer(self, key, segment_size, sink=None, resume_from=None):
        # stub: ignores segment_size validation, resume, and integrity re-fetch
        data = self._cold.get(key)
        if data is None:
            return None
        idx = 0
        for off in range(0, len(data), max(1, segment_size)):
            seg = data[off:off + segment_size]
            if sink is not None:
                sink.write_segment(idx, seg, zlib.crc32(seg) & 0xFFFFFFFF)
            idx += 1
        return idx
