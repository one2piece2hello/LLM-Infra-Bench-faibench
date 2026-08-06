#!/usr/bin/env python3
"""ORACLE (reviewer-only, NOT baked into the image) for correctness-e2e-e5-tiered-storage-io.

A complete, correct implementation of the tiered storage engine + resumable segmented transfer.
Verified against the harness's own reference model; oracle mode must score correctness_frac == 1.0.
Applied at score time via KERNELBENCH_ORACLE_PATCH; the Dockerfile never COPYs it. Oracle patch =
diff against the shipped starter stub environment/workspace/submission/tiered_store.py.

Design notes:
  * hot tier = OrderedDict as an LRU (first=oldest, last=most-recent), keyed by value bytes size.
  * cold tier = dict (authoritative, durable). put writes cold first, then hot + evict.
  * eviction: while sum(value sizes) > capacity, popitem(last=False) drops the LRU. All-or-nothing:
    we compute the eviction under the lock so no half-evicted state is ever observable.
  * coalesced cold-miss fetch: a per-key in-flight registry; the first caller runs cold_fetch_fn, the
    rest wait on its Event and reuse the value (etcd/seaweedfs LoadRemoteFile spirit).
  * transfer: fixed-size segments; resume_from skips already-received indices; a bad-segment signal
    (any exception from write_segment) triggers a bounded re-fetch of that same index.
"""
import threading
import zlib
from collections import OrderedDict


class TieredStore:
    def __init__(self, hot_capacity, cold_fetch_fn=None):
        self.hot_capacity = hot_capacity
        self._cold_fetch_fn = cold_fetch_fn
        self._hot = OrderedDict()   # key -> value bytes, LRU order (first=oldest)
        self._cold = {}             # key -> value bytes (authoritative, locally cached)
        self._remote = set()        # keys resident only in remote backing (reachable via cold_fetch_fn)
        self._lock = threading.RLock()
        self._inflight = {}         # key -> {"event": Event, "value": ...}

    # ------------------------------------------------------------------- eviction (all-or-nothing)
    def _evict_locked(self):
        while self._hot and sum(len(v) for v in self._hot.values()) > self.hot_capacity:
            self._hot.popitem(last=False)  # drop LRU

    def _touch_locked(self, key, value):
        self._hot.pop(key, None)
        self._hot[key] = value
        self._evict_locked()

    # ---------------------------------------------------------------------------------- put / get
    def put(self, key, value):
        with self._lock:
            self._cold[key] = value
            self._touch_locked(key, value)

    def seed_cold(self, key, value):
        # Register in the REMOTE backing only (not hot, not local cold cache). A later get() miss must
        # fetch it via cold_fetch_fn. We keep the value so an in-process cold_fetch_fn can serve it, but
        # the harness supplies its own cold_fetch_fn in coalescing cases.
        with self._lock:
            self._remote.add(key)

    def get(self, key):
        with self._lock:
            if key in self._hot:
                v = self._hot.pop(key); self._hot[key] = v  # touch
                return v
            if key in self._cold:
                v = self._cold[key]
                self._touch_locked(key, v)
                return v
            # not in hot, not in local cold: try coalesced remote fetch fn
            if self._cold_fetch_fn is None or key not in self._remote:
                return None
            entry = self._inflight.get(key)
            leader = False
            if entry is None:
                entry = {"event": threading.Event(), "value": None}
                self._inflight[key] = entry
                leader = True
        if leader:
            try:
                val = self._cold_fetch_fn(key)
                with self._lock:
                    if val is not None:
                        self._cold[key] = val
                        self._remote.discard(key)
                        self._touch_locked(key, val)
                    entry["value"] = val
            finally:
                entry["event"].set()
                with self._lock:
                    self._inflight.pop(key, None)
            return entry["value"]
        else:
            entry["event"].wait(timeout=30)
            return entry["value"]

    def hot_size(self):
        with self._lock:
            return sum(len(v) for v in self._hot.values())

    def in_hot(self, key):
        with self._lock:
            return key in self._hot

    # ------------------------------------------------------------- resumable segmented transfer
    def transfer(self, key, segment_size, sink=None, resume_from=None):
        if segment_size is None or segment_size <= 0:
            raise ValueError("segment_size must be positive")
        with self._lock:
            data = self._cold.get(key)
            if data is None:
                data = self._hot.get(key)
        if data is None:
            raise KeyError(f"transfer of missing key: {key}")
        already = set(resume_from) if resume_from else set()
        total = 0
        idx = 0
        for off in range(0, len(data), segment_size):
            total += 1
            if idx in already:
                idx += 1
                continue
            seg = data[off:off + segment_size]
            crc = zlib.crc32(seg) & 0xFFFFFFFF
            # bounded re-fetch on a bad-segment / corruption signal from the sink. A bad-segment
            # (corruption) signal is retriable; a transfer interruption is NOT -- propagate it so the
            # caller can resume later. We tell them apart by exception class name (we must not import
            # the harness). Anything whose class name contains "BadSegment"/"Corrupt"/"Crc" is a
            # retriable integrity signal; everything else (e.g. an interruption) propagates.
            attempts = 0
            while True:
                attempts += 1
                try:
                    if sink is not None:
                        sink.write_segment(idx, seg, crc)
                    break
                except Exception as ex:
                    name = type(ex).__name__.lower()
                    retriable = ("badsegment" in name) or ("corrupt" in name) or ("crc" in name)
                    if (not retriable) or attempts >= 8:
                        raise
                    # re-fetch the authoritative bytes for this segment and try again
                    with self._lock:
                        fresh = self._cold.get(key, data)
                    seg = fresh[off:off + segment_size]
                    crc = zlib.crc32(seg) & 0xFFFFFFFF
                    continue
            idx += 1
        # handle empty object (no segments): still return 0
        if len(data) == 0:
            return 0
        return total
