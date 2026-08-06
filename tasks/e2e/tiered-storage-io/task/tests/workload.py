#!/usr/bin/env python3
"""correctness-e2e-e5-tiered-storage-io harness (reviewer-authored, uploaded with tests/).

IMPL-CLASS correctness task (perf_metric:none). reward is BINARY (reward.md 实现类): 1.0 iff EVERY
graded case passes and no cheat/hard-fail gate trips, else 0.0. passed/total and the per-axis
breakdown are still emitted for offline diagnosis but NEVER scale the reward. Cases run over a
LARGE graded hidden case set. The candidate implements a tiered storage engine with a
hot in-memory tier and a cold "remote" tier, LRU/size-driven eviction, resumable segmented transfer
of cold objects, and cross-tier read consistency, from a frozen contract. This harness owns the
reference model and grades the candidate along six axes.

--------------------------------------------------------------------------------------------------
Provenance (real atoms, medium-topic E5 CROSS.STORAGE.TIERED / TRANSFER / CONSISTENCY):
  * seaweedfs weed/storage/volume_tier.go LoadRemoteFile -- volume-tier metadata records remote key /
    offset / size, and when local data is missing it loads the remote file to keep serving. (our
    tier-miss-fetch + tier-metadata axes)
  * apache/rocketmq TieredMessageStore -- coordinates reads between a local commitlog and a remote
    tier. (our read-through / write-back tiering axes)
  * apache/kafka RemoteLogManager -- manages tiered log-segment upload, metadata and remote read.
    (our segment upload + metadata axes)
  * etcd kvstore_compaction.go store.scheduleCompaction -- updates the in-memory index first, then
    deletes historical revisions in batches with a forced commit, and only writes the finished
    revision when ALL batches complete. (our batched-eviction + all-or-nothing-finish axes)
  * minio cmd/erasure-coding.go -- Reed-Solomon-style encode: reconstruct the object byte stream when
    some shards are missing. (our resumable/partial-transfer integrity axis, simplified to checksum)
  * redis/valkey aof.c multi-segment manifest -- base / incremental / history segments, atomic
    manifest switch after rewrite. (our resumable-transfer manifest + atomic-swap axes)

Contract under test (submission/tiered_store.py). The candidate implements a single class
`TieredStore` exposing put / get / a size-bounded hot tier with eviction to a cold tier, a resumable
segmented `transfer` of a cold object back into the hot tier, and cross-tier consistency guarantees.
The harness drives it deterministically (single-threaded cases) AND under real threads (concurrency
cases), grading every case against a harness-owned reference model. Model-free at import.

Scoring: reward is BINARY -- 1.0 iff EVERY graded case passes and no cheat/gate trips, else 0.0 (passed/total never scales it); hard-fail gates in test.sh force 0.0.
Per-axis pass rate emitted for the reviewer.
"""
import json
import os
import sys
import importlib.util
import threading
import traceback

SUB = os.environ.get("E2E_SUBMISSION_DIR", "/app/submission")
ENTRY = os.path.join(SUB, "tiered_store.py")


def _load_candidate():
    if not os.path.isfile(ENTRY):
        raise FileNotFoundError(f"entry missing: {ENTRY}")
    spec = importlib.util.spec_from_file_location("candidate_ts", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- harness-owned reference model
# A dead-simple, obviously-correct reference for a size-bounded LRU hot tier over a cold backing
# store. The candidate's job is to reproduce these semantics (with eviction, resumable transfer and
# cross-tier consistency) -- NOT to be its own oracle.

def _blob(seed, size):
    """Deterministic pseudo-random bytes for value payloads (harness-owned, reproducible)."""
    out = bytearray(size)
    x = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    for i in range(size):
        x = (1103515245 * x + 12345) & 0xFFFFFFFF
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


def _ref_lru_capacity(ops, hot_capacity):
    """Reference: replay put/get ops against a size-bounded (by value bytes) LRU hot tier over an
    unbounded cold tier. Returns (final_cold_state, final_hot_keys_set, get_results).
    Semantics: put(k,v) inserts into hot (touch=most-recent) and also durably records into cold;
    get(k) returns v from hot if present (touch) else fetches from cold into hot (touch, may evict).
    Eviction: while sum(hot value sizes) > hot_capacity, evict the LEAST-recently-used key from hot
    (its authoritative copy remains in cold). A missing key returns None."""
    from collections import OrderedDict
    hot = OrderedDict()   # key -> value bytes, ordered LRU (first=oldest)
    cold = {}             # key -> value bytes (authoritative, durable)
    get_results = []

    def _evict():
        while sum(len(v) for v in hot.values()) > hot_capacity and hot:
            hot.popitem(last=False)  # drop LRU

    for op in ops:
        if op[0] == "put":
            _, k, v = op
            cold[k] = v
            hot.pop(k, None)
            hot[k] = v         # most-recent
            _evict()
        elif op[0] == "get":
            _, k = op
            if k in hot:
                v = hot.pop(k); hot[k] = v  # touch
                get_results.append((k, v))
            elif k in cold:
                v = cold[k]
                hot[k] = v; _evict()        # fetch cold->hot
                get_results.append((k, v))
            else:
                get_results.append((k, None))
        else:
            raise ValueError(f"bad ref op {op[0]}")
    return cold, set(hot.keys()), get_results


def _crc32(b):
    import zlib
    return zlib.crc32(b) & 0xFFFFFFFF


# --------------------------------------------------------------------------------- case catalogue

def _build_cases():
    cases = []
    cid = 0

    def add(axis, kind, note, **kw):
        nonlocal cid
        c = {"cid": cid, "axis": axis, "kind": kind, "note": note}
        c.update(kw)
        cases.append(c)
        cid += 1

    # ---- NORMAL: basic put/get, hot-hit, cold-fetch, get-missing ----
    add("normal", "put_get", "single put then get returns value",
        ops=[("put", "a", _blob(1, 32)), ("get", "a")], hot_capacity=1 << 20)
    add("normal", "put_get", "get missing key returns None",
        ops=[("put", "a", _blob(1, 16)), ("get", "z")], hot_capacity=1 << 20)
    add("normal", "put_get", "overwrite: last put wins",
        ops=[("put", "a", _blob(1, 16)), ("put", "a", _blob(2, 16)), ("get", "a")], hot_capacity=1 << 20)
    add("normal", "tier_meta", "cold tier is authoritative after put",
        ops=[("put", "k1", _blob(3, 64)), ("put", "k2", _blob(4, 64))], hot_capacity=1 << 20)

    # ---- BOUNDARY: empty store, zero-length value, exactly-at-capacity, single-slot tier ----
    add("boundary", "put_get", "empty store: get returns None", ops=[("get", "a")], hot_capacity=1 << 20)
    add("boundary", "put_get", "zero-length value round-trips",
        ops=[("put", "e", b""), ("get", "e")], hot_capacity=1 << 20)
    add("boundary", "eviction", "value exactly at capacity: stays resident",
        ops=[("put", "a", _blob(1, 100)), ("get", "a")], hot_capacity=100, check_hot_set=True)
    add("boundary", "eviction", "value larger than capacity still readable via cold",
        ops=[("put", "big", _blob(9, 256)), ("get", "big")], hot_capacity=128)
    add("boundary", "put_get", "many small keys",
        ops=[("put", f"k{i}", _blob(i, 8)) for i in range(200)] + [("get", f"k{i}") for i in range(0, 200, 37)],
        hot_capacity=1 << 20)

    # ---- DEGENERATE: repeated puts to same key, get same key repeatedly, all-same-size churn ----
    add("degenerate", "put_get", "repeated puts same key (last wins), then get",
        ops=[("put", "a", _blob(i, 16)) for i in range(20)] + [("get", "a")], hot_capacity=1 << 20)
    add("degenerate", "eviction", "churn > capacity forces LRU eviction to cold, all still readable",
        ops=[("put", f"k{i}", _blob(i, 64)) for i in range(50)] + [("get", f"k{i}") for i in range(50)],
        hot_capacity=64 * 8)
    add("degenerate", "eviction", "repeated get of one key keeps it MRU across interleaved gets",
        ops=([("put", f"k{i}", _blob(i, 64)) for i in range(6)]
             + [("get", "k0"), ("get", "k1"), ("get", "k0"), ("get", "k1"), ("get", "k0")]),
        hot_capacity=64 * 3, check_hot_set=True)

    # ---- ERROR: bad op / bad transfer parameters must raise or be rejected cleanly ----
    add("error", "bad_op", "unknown store op must raise",
        ops=[("frobnicate", "a", b"x")], hot_capacity=1 << 20)
    add("error", "transfer_bad", "transfer of a non-existent key must raise or return None",
        put_key="a", put_val=_blob(1, 128), transfer_key="ghost", seg=32)
    add("error", "transfer_bad_seg", "transfer with non-positive segment size must raise",
        put_key="a", put_val=_blob(1, 128), transfer_key="a", seg=0)

    # ---- METAMORPHIC: idempotent re-put, get-does-not-mutate-value, capacity-invariance of values ----
    add("metamorphic", "idempotent_reput", "re-putting same key/value is idempotent (cold unchanged)",
        ops=[("put", "a", _blob(1, 64)), ("put", "a", _blob(1, 64)), ("get", "a")], hot_capacity=1 << 20)
    add("metamorphic", "value_invariance", "values survive eviction+refetch unchanged (checksum)",
        ops=[("put", f"k{i}", _blob(i, 64)) for i in range(30)] + [("get", f"k{i}") for i in range(30)],
        hot_capacity=64 * 4)
    add("metamorphic", "order_invariance", "get results equal reference regardless of hot capacity",
        ops=[("put", "a", _blob(1, 32)), ("put", "b", _blob(2, 32)), ("get", "a"), ("get", "b")],
        hot_capacity=48)

    # ---- HIDDEN-REGIME: resumable transfer + integrity + concurrency (the hard part) ----
    #  (a) resumable segmented transfer: fetch a cold object in fixed-size segments; a transfer that
    #      is interrupted after k segments and RESUMED must yield the identical object (checksum),
    #      and must not re-send already-received segments (manifest-driven resume, AOF/kafka-style).
    add("hidden_regime", "resumable_transfer", "interrupted transfer resumes to identical object",
        put_key="obj", put_val=_blob(7, 1000), seg=128, interrupt_after=3)
    #  (b) transfer integrity: a corrupted segment must be detected (crc mismatch) and re-fetched,
    #      never silently accepted (minio-reconstruct spirit, simplified to checksum verification).
    add("hidden_regime", "transfer_integrity", "corrupted segment is rejected and re-fetched",
        put_key="obj", put_val=_blob(8, 640), seg=128, corrupt_segment=2)
    #  (c) concurrent readers during eviction: while a writer churns the hot tier, concurrent readers
    #      must NEVER observe a torn or phantom value; every observed value must be a value actually
    #      put, and the cold tier must equal the reference after all writes.
    add("hidden_regime", "concurrent_rw", "concurrent readers see no torn/phantom values under churn",
        n_keys=40, val_size=48, hot_capacity=48 * 6, n_reader_threads=8)
    #  (d) coalesced tier-miss fetch: many threads get() the same cold-only key at once; the store
    #      should fetch it from cold at most a bounded number of times (coalesce), and all callers
    #      must observe the identical value (seaweedfs LoadRemoteFile on local miss).
    add("hidden_regime", "coalesce_fetch", "concurrent cold-miss fetches coalesce, agree on value",
        key="cold1", val=_blob(11, 96), hot_capacity=96 * 2, n_threads=16, max_fetches=12)
    #  (e) all-or-nothing batched eviction commit: an eviction batch either fully commits (keys gone
    #      from hot, present in cold) or leaves the tier untouched -- never a half-evicted state
    #      (etcd scheduleCompaction finished-revision spirit).
    add("hidden_regime", "atomic_eviction", "batched eviction is all-or-nothing (no half state)",
        ops=[("put", f"k{i}", _blob(i, 64)) for i in range(12)], hot_capacity=64 * 3)

    return cases


# --------------------------------------------------------------------------------------- graders

def _grade(case, mod):
    axis, kind = case["axis"], case["kind"]
    try:
        # ---------- bad op ----------
        if kind == "bad_op":
            ts = mod.TieredStore(hot_capacity=case["hot_capacity"])
            try:
                _replay(ts, case["ops"])
                return False, "bad_op_did_not_raise"
            except Exception:
                return True, "bad_op_raised"

        # ---------- put/get, tier_meta, idempotent, invariance, order ----------
        if kind in ("put_get", "tier_meta", "idempotent_reput", "value_invariance", "order_invariance"):
            ts = mod.TieredStore(hot_capacity=case["hot_capacity"])
            got = _replay(ts, case["ops"])
            ref_cold, _ref_hot, ref_gets = _ref_lru_capacity(case["ops"], case["hot_capacity"])
            # every get result must match the reference
            if got != ref_gets:
                return False, f"get_mismatch got={_short(got)} exp={_short(ref_gets)}"
            # cold tier is authoritative: every put key must be retrievable and byte-identical
            for k, v in ref_cold.items():
                cg = ts.get(k)
                if cg != v:
                    return False, f"cold_read[{k}] mismatch len_got={_ln(cg)} len_exp={len(v)}"
            return True, f"{kind}_ok"

        # ---------- eviction: values must survive eviction and be byte-identical on refetch ----------
        if kind == "eviction":
            ts = mod.TieredStore(hot_capacity=case["hot_capacity"])
            got = _replay(ts, case["ops"])
            ref_cold, ref_hot, ref_gets = _ref_lru_capacity(case["ops"], case["hot_capacity"])
            if got != ref_gets:
                return False, f"evict_get_mismatch got={_short(got)} exp={_short(ref_gets)}"
            # hot-tier size invariant. hot_size() is a REQUIRED part of the published contract
            # (instruction.md), so a missing / non-callable hot_size is itself a failure: it must
            # never be able to silently switch the capacity assertion off.
            if not callable(getattr(ts, "hot_size", None)):
                return False, "hot_size() missing or not callable (required by the contract)"
            if ts.hot_size() > case["hot_capacity"]:
                return False, f"hot_size {ts.hot_size()} > cap {case['hot_capacity']}"
            # Residency: WHICH keys the size-driven LRU policy leaves resident is part of the
            # contract (strict `>` eviction boundary + evict-least-recently-used). Expected set
            # comes from the harness reference model, never hand-computed. Only cases that ask.
            if case.get("check_hot_set"):
                if not callable(getattr(ts, "in_hot", None)):
                    return False, "in_hot() missing or not callable (required by the contract)"
                got_hot = {k for k in ref_cold if ts.in_hot(k)}
                if got_hot != ref_hot:
                    return False, (f"hot residency {sorted(got_hot)} != reference "
                                   f"{sorted(ref_hot)} (LRU recency / eviction boundary)")
            # all put keys still readable byte-identical
            for k, v in ref_cold.items():
                if ts.get(k) != v:
                    return False, f"post_evict_read[{k}] mismatch"
            return True, "eviction_ok"

        # ---------- transfer bad params ----------
        if kind == "transfer_bad":
            ts = mod.TieredStore(hot_capacity=1 << 20)
            ts.put(case["put_key"], case["put_val"])
            try:
                r = ts.transfer(case["transfer_key"], segment_size=case["seg"])
                if r is None:
                    return True, "transfer_missing_returned_none"
                return False, "transfer_of_missing_key_did_not_signal"
            except Exception:
                return True, "transfer_missing_raised"

        if kind == "transfer_bad_seg":
            ts = mod.TieredStore(hot_capacity=1 << 20)
            ts.put(case["put_key"], case["put_val"])
            try:
                ts.transfer(case["transfer_key"], segment_size=case["seg"])
                return False, "nonpositive_seg_not_rejected"
            except Exception:
                return True, "nonpositive_seg_raised"

        # ---------- resumable transfer ----------
        if kind == "resumable_transfer":
            ts = mod.TieredStore(hot_capacity=1 << 20)
            ts.put(case["put_key"], case["put_val"])
            # force the key to be cold-only (evict via a tiny capacity refetch is optional); use
            # transfer directly which fetches from cold in segments.
            sink = _SegSink(interrupt_after=case["interrupt_after"])
            # first pass: interrupted after k segments
            try:
                ts.transfer(case["put_key"], segment_size=case["seg"], sink=sink)
            except _Interrupted:
                pass
            partial_segs = sink.n_received()
            if partial_segs < 1:
                return False, "no segments received before interrupt"
            # resume: sink already holds the manifest of received segments; must NOT resend them
            sink2 = sink.resume()
            ts.transfer(case["put_key"], segment_size=case["seg"], sink=sink2, resume_from=sink.manifest())
            data = sink2.assembled()
            if _crc32(data) != _crc32(case["put_val"]):
                return False, "resumed object checksum mismatch"
            # resume must not have re-sent already-received segments
            if sink2.resent_already_received():
                return False, "resume re-sent already-received segments"
            return True, f"resumable_ok segs={sink2.n_received_total()}"

        # ---------- transfer integrity (corruption detection) ----------
        if kind == "transfer_integrity":
            ts = mod.TieredStore(hot_capacity=1 << 20)
            ts.put(case["put_key"], case["put_val"])
            sink = _SegSink(corrupt_segment=case["corrupt_segment"])
            ts.transfer(case["put_key"], segment_size=case["seg"], sink=sink)
            data = sink.assembled()
            if _crc32(data) != _crc32(case["put_val"]):
                return False, "corruption not detected/repaired: checksum mismatch"
            if not sink.refetched_corrupt():
                return False, "corrupt segment accepted without re-fetch"
            return True, "integrity_ok"

        # ---------- concurrent readers/writer under churn ----------
        if kind == "concurrent_rw":
            ts = mod.TieredStore(hot_capacity=case["hot_capacity"])
            vals = {f"k{i}": _blob(i, case["val_size"]) for i in range(case["n_keys"])}
            valset = set(vals.values())
            observed = []
            rd_errs = []
            olk = threading.Lock()
            done = threading.Event()

            def writer():
                for i in range(case["n_keys"]):
                    ts.put(f"k{i}", vals[f"k{i}"])
                done.set()

            def reader():
                # One FULL pass is always made after the writer is done, so a compliant store
                # always yields observations; a reader that raises is recorded, never swallowed.
                keys = list(vals.keys())
                while True:
                    last = done.is_set()
                    for k in keys:
                        try:
                            v = ts.get(k)
                        except Exception as e:
                            with olk:
                                rd_errs.append(f"{type(e).__name__}:{e}")
                            return
                        if v is not None:
                            with olk:
                                observed.append(v)
                    if last:
                        return

            wt = threading.Thread(target=writer)
            rts = [threading.Thread(target=reader) for _ in range(case["n_reader_threads"])]
            wt.start()
            for t in rts:
                t.start()
            wt.join(timeout=30)
            for t in rts:
                t.join(timeout=30)
            if wt.is_alive() or any(t.is_alive() for t in rts):
                return False, "concurrent_rw hung"
            # a concurrent get() that raises is a contract violation, not an excuse
            if rd_errs:
                return False, f"reader thread raised under concurrency: {rd_errs[:3]}"
            if not observed:
                return False, "no concurrent read observed any value (readers never saw the store)"
            # no torn/phantom values
            bad = [v for v in observed if v not in valset]
            if bad:
                return False, f"phantom/torn values observed ({len(bad)})"
            # final: every key readable byte-identical
            for k, v in vals.items():
                if ts.get(k) != v:
                    return False, f"final read[{k}] mismatch"
            return True, f"concurrent_rw_ok observed={len(observed)}"

        # ---------- coalesced cold-miss fetch ----------
        if kind == "coalesce_fetch":
            fetches = {"n": 0}
            flk = threading.Lock()

            def counting_cold_fetch(k):
                with flk:
                    fetches["n"] += 1
                return case["val"]

            ts = mod.TieredStore(hot_capacity=case["hot_capacity"], cold_fetch_fn=counting_cold_fetch)
            # register the key as cold-only (present in cold backing, absent from hot)
            ts.seed_cold(case["key"], case["val"])
            results = [None] * case["n_threads"]
            barrier = threading.Barrier(case["n_threads"])

            def rd(i):
                barrier.wait()
                results[i] = ts.get(case["key"])

            ths = [threading.Thread(target=rd, args=(i,)) for i in range(case["n_threads"])]
            for t in ths:
                t.start()
            for t in ths:
                t.join(timeout=30)
            if any(t.is_alive() for t in ths):
                return False, "coalesce_fetch hung"
            if any(r != case["val"] for r in results):
                return False, "coalesced readers disagree on value"
            if fetches["n"] > case["max_fetches"]:
                return False, f"too many cold fetches {fetches['n']} > {case['max_fetches']}"
            if fetches["n"] < 1:
                return False, "no cold fetch issued (value fabricated?)"
            return True, f"coalesce_ok fetches={fetches['n']}"

        # ---------- atomic batched eviction ----------
        if kind == "atomic_eviction":
            ts = mod.TieredStore(hot_capacity=case["hot_capacity"])
            _replay(ts, case["ops"])
            ref_cold, ref_hot, _ = _ref_lru_capacity(case["ops"], case["hot_capacity"])
            # every put key must be present in cold (durable) and byte-identical
            for k, v in ref_cold.items():
                if ts.get(k) != v:
                    return False, f"post_evict cold read[{k}] mismatch"
            # hot-size invariant respected (no half-evicted overflow)
            if not callable(getattr(ts, "hot_size", None)):
                return False, "hot_size() missing or not callable (required by the contract)"
            if ts.hot_size() > case["hot_capacity"]:
                return False, f"hot_size {ts.hot_size()} > cap {case['hot_capacity']} (half-evicted)"
            return True, "atomic_eviction_ok"

    except Exception as e:
        return False, f"exc:{type(e).__name__}:{e}"
    return False, f"unknown_kind:{kind}"


# ---- helpers shared by graders -----------------------------------------------------------------

def _replay(ts, ops):
    """Drive put/get ops; returns list of (key, value) for each get."""
    out = []
    for op in ops:
        if op[0] == "put":
            ts.put(op[1], op[2])
        elif op[0] == "get":
            out.append((op[1], ts.get(op[1])))
        else:
            raise ValueError(f"bad op {op[0]}")
    return out


def _short(gets):
    return [(k, (v[:4] + b"..") if isinstance(v, (bytes, bytearray)) and len(v) > 4 else v) for k, v in gets][:6]


def _ln(v):
    return len(v) if isinstance(v, (bytes, bytearray)) else None


class _Interrupted(Exception):
    pass


class _SegSink:
    """Harness-owned sink the candidate's transfer() writes segments into. Simulates an interruptible,
    integrity-checked segmented transfer channel. The candidate must:
      - write each segment as write_segment(index, data, crc);
      - on resume (resume_from manifest given), NOT re-send indices already in the manifest;
      - on a crc/corruption signal, re-fetch that segment (we flip corrupt once, then serve clean)."""

    def __init__(self, interrupt_after=None, corrupt_segment=None, _carry=None):
        self.interrupt_after = interrupt_after
        self.corrupt_segment = corrupt_segment
        self._segs = dict(_carry) if _carry else {}   # index -> data
        self._order = []                                # indices in receive order
        self._resent = []                               # indices re-sent during resume
        self._prior = set(_carry.keys()) if _carry else set()
        self._corrupt_served = set()
        self._refetched = set()

    def write_segment(self, index, data, crc=None):
        # integrity: on the designated segment, reject the first delivery (bad crc) once.
        if self.corrupt_segment is not None and index == self.corrupt_segment \
                and index not in self._corrupt_served:
            self._corrupt_served.add(index)
            raise _BadSegment(index)   # signals candidate to re-fetch this segment
        if index in self._prior:
            self._resent.append(index)
        if self.corrupt_segment is not None and index in self._corrupt_served \
                and index not in self._segs:
            self._refetched.add(index)
        self._segs[index] = data
        self._order.append(index)
        if self.interrupt_after is not None and len(self._order) >= self.interrupt_after \
                and not self._prior:
            raise _Interrupted()
        return True

    def manifest(self):
        return sorted(self._segs.keys())

    def n_received(self):
        return len([i for i in self._order if i not in self._prior])

    def n_received_total(self):
        return len(self._segs)

    def resume(self):
        return _SegSink(interrupt_after=None, corrupt_segment=self.corrupt_segment, _carry=self._segs)

    def resent_already_received(self):
        return len(self._resent) > 0

    def refetched_corrupt(self):
        return len(self._refetched) > 0 or len(self._corrupt_served) > 0

    def assembled(self):
        return b"".join(self._segs[i] for i in sorted(self._segs.keys()))


class _BadSegment(Exception):
    def __init__(self, index):
        super().__init__(f"bad segment {index}")
        self.index = index


def run_cases():
    try:
        mod = _load_candidate()
    except Exception as e:
        return {"completed": False, "all_passed": False, "reward_binary": 0.0,
                "tests": {"passed": 0, "total": 0}, "passed": 0, "total": 0,
                "correctness_frac": 0.0, "error": f"{type(e).__name__}: {e}"}
    cases = _build_cases()
    results = []
    by_axis = {}
    passed = 0
    for case in cases:
        try:
            ok, reason = _grade(case, mod)
        except Exception as e:
            ok, reason = False, f"harness_exc:{type(e).__name__}:{e}"
        results.append({"cid": case["cid"], "axis": case["axis"], "kind": case["kind"],
                        "passed": bool(ok), "reason": reason, "note": case["note"]})
        a = by_axis.setdefault(case["axis"], {"passed": 0, "total": 0})
        a["total"] += 1
        if ok:
            a["passed"] += 1
            passed += 1
    total = len(cases)
    all_passed = bool(total > 0 and passed == total)
    return {"completed": True, "expected_case_count": total, "total": total, "passed": passed,
            # ---- BINARY aggregation (reward.md 实现类): all cases pass -> 1.0, any fail -> 0.0 ----
            "all_passed": all_passed, "reward_binary": 1.0 if all_passed else 0.0,
            "tests": {"passed": passed, "total": total},
            # ---- diagnostics only (MUST NOT influence reward) ----
            "correctness_frac": (passed / total) if total else 0.0, "by_axis": by_axis, "cases": results}


def main():
    try:
        trace = run_cases()
    except Exception as e:
        trace = {"completed": False, "all_passed": False, "reward_binary": 0.0,
                 "tests": {"passed": 0, "total": 0}, "passed": 0, "total": 0,
                 "correctness_frac": 0.0,
                 "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
    print("E2E_RESULT " + json.dumps({"correctness_trace": trace}))


if __name__ == "__main__":
    main()
