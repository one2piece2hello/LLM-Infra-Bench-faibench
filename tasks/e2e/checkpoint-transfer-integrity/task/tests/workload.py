#!/usr/bin/env python3
"""correctness-e2e-e5-checkpoint-transfer-integrity harness (reviewer-authored, uploaded with tests/).

IMPL-CLASS correctness task (perf_metric:none). reward is BINARY (reward.md 实现类): 1.0 iff EVERY
graded case passes and no cheat/hard-fail gate trips, else 0.0. passed/total and the per-axis
breakdown are still emitted for offline diagnosis but NEVER scale the reward. Cases run over a
LARGE graded hidden case set. The candidate implements a checkpoint-transfer manager
that shards a checkpoint blob into fixed-size chunks with per-chunk checksums + a manifest, performs
an all-or-nothing durable upload, verifies integrity on download, RESUMES partial transfers from the
manifest, reconstructs the blob from XOR parity when a bounded number of chunks are missing, and
normalizes / de-conflicts artifact paths. This harness owns the reference and grades six axes.

--------------------------------------------------------------------------------------------------
Provenance (real atoms, medium-topic E5 CROSS.STORAGE.TRANSFER / CHECKPOINT / INTEGRITY):
  * NVIDIA/NeMo s3_checkpoint_io.py S3CheckpointIO -- maps a checkpoint save onto object storage and
    hides remote IO latency with async upload / local staging. (our staged upload + durability axes)
  * harbor trial/artifact_handler.py ArtifactHandler -- scans an artifacts manifest, handles path
    normalization / conflicts / stepwise download, and uploads results to persistent storage.
    (our manifest + path-normalization + conflict axes)
  * ceph crimson ECBackend.submit_transaction -- launches per-shard erasure-coded subwrites and only
    completes the durability future once ALL subwrites commit. (our all-or-nothing durable-commit axis)
  * minio cmd/erasure-coding.go -- Reed-Solomon data+parity shards; reconstruct the object byte stream
    when some shards are missing. (our XOR-parity reconstruction axis, simplified to single-parity)
  * redis/valkey aof.c multi-segment manifest -- base / incremental / history segments with an atomic
    manifest switch. (our manifest + resumable-download + atomic-finalize axes)
  * EleutherAI/gpt-neox megatron/checkpointing.py -- unified checkpoint tag, retained dirs, old
    checkpoint cleanup for S3 saves. (our finalize / cleanup-on-failure axis)

Contract under test (submission/ckpt_transfer.py). The candidate implements a single class
`CheckpointTransfer` exposing pack (blob -> chunks+parity+manifest), a durable all-or-nothing upload
through an injected chunk store, a verifying download that resumes from a manifest and reconstructs
from parity, plus artifact-path normalization and conflict detection. The harness drives it
deterministically AND under real threads, grading every case against a harness-owned reference model.

Scoring: reward = passed / total_cases (float in [0,1]); hard-fail gates in test.sh force 0.0.
Per-axis pass rate emitted for the reviewer.
"""
import json
import os
import sys
import importlib.util
import threading
import traceback
import zlib

SUB = os.environ.get("E2E_SUBMISSION_DIR", "/app/submission")
ENTRY = os.path.join(SUB, "ckpt_transfer.py")


def _load_candidate():
    if not os.path.isfile(ENTRY):
        raise FileNotFoundError(f"entry missing: {ENTRY}")
    spec = importlib.util.spec_from_file_location("candidate_ct", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- harness-owned reference model
def _blob(seed, size):
    out = bytearray(size)
    x = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    for i in range(size):
        x = (1103515245 * x + 12345) & 0xFFFFFFFF
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


def _crc32(b):
    return zlib.crc32(b) & 0xFFFFFFFF


def _ref_chunks(blob, chunk_size):
    """Reference: split blob into fixed-size data chunks (last may be short). Returns list of bytes."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [blob[i:i + chunk_size] for i in range(0, len(blob), chunk_size)] or [b""]


def _ref_parity(data_chunks, chunk_size):
    """Single XOR parity across padded data chunks (minio/EC spirit, single-parity variant).
    Returns a parity chunk of length chunk_size that lets ONE missing data chunk be reconstructed."""
    par = bytearray(chunk_size)
    for c in data_chunks:
        for i in range(len(c)):
            par[i] ^= c[i]
    return bytes(par)


def _ref_normalize(path):
    """Reference artifact-path normalization: collapse '.', '..', duplicate '/', strip trailing '/'.
    Rejects (returns None) any path that escapes the root (starts with '..' after normalization)."""
    parts = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None  # escapes root
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


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

    # ---- NORMAL: pack->manifest, upload+download round-trip, checksum ----
    add("normal", "roundtrip", "pack + upload + download == original blob",
        blob=_blob(1, 1000), chunk_size=128)
    add("normal", "manifest", "manifest lists correct chunk count + total size + crc",
        blob=_blob(2, 640), chunk_size=128)
    add("normal", "roundtrip", "exact multiple of chunk_size",
        blob=_blob(3, 512), chunk_size=128)

    # ---- BOUNDARY: empty blob, single byte, blob < chunk_size, one huge chunk ----
    add("boundary", "roundtrip", "empty blob round-trips", blob=b"", chunk_size=64)
    add("boundary", "roundtrip", "single byte blob", blob=b"\x07", chunk_size=64)
    add("boundary", "roundtrip", "blob smaller than chunk_size (one short chunk)",
        blob=_blob(4, 40), chunk_size=128)
    add("boundary", "roundtrip", "chunk_size larger than needed, many chunks",
        blob=_blob(5, 2048), chunk_size=64)

    # ---- DEGENERATE: all-zero blob, repeated identical chunks, tiny chunk_size ----
    add("degenerate", "roundtrip", "all-zero blob round-trips", blob=bytes(600), chunk_size=100)
    add("degenerate", "roundtrip", "repeated identical chunks",
        blob=b"ABCD" * 200, chunk_size=8)
    add("degenerate", "roundtrip", "chunk_size == 1 (many 1-byte chunks)",
        blob=_blob(6, 64), chunk_size=1)

    # ---- ERROR: bad chunk_size, path escaping root, download with tampered manifest crc ----
    # 2026-07-27: this used to pass chunk_size=0. `range(0, len(blob), 0)` raises ValueError all by
    # itself, so an implementation with NO chunk_size validation passed for free, while a NEGATIVE
    # chunk_size -- which such an implementation accepts silently (an empty range, or max(1, cs)) --
    # was never tested. -4 strictly dominates 0: the oracle rejects "chunk_size <= 0" either way,
    # but only a negative value distinguishes real validation from an incidental interpreter error.
    add("error", "bad_chunk_size", "chunk_size <= 0 must raise", blob=_blob(7, 100), chunk_size=-4)
    add("error", "path_escape", "'..' escaping root is rejected", path="a/../../etc/passwd", expect=None)
    add("error", "tamper_detect", "download must FAIL (raise/None) when a chunk AND its parity are corrupt",
        blob=_blob(8, 512), chunk_size=128, tamper_chunk=1, tamper_parity=True)

    # ---- METAMORPHIC: pack is deterministic, path normalization idempotent, upload idempotent ----
    add("metamorphic", "deterministic_pack", "packing the same blob twice yields identical manifest",
        blob=_blob(9, 777), chunk_size=100)
    add("metamorphic", "normalize_idempotent", "normalize(normalize(p)) == normalize(p)",
        path="./a//b/../c/./d/")
    add("metamorphic", "normalize_paths", "a batch of paths normalize as reference",
        paths=["a/b/c", "a//b/./c", "a/b/x/../c", "/a/b/c/", "x/../a/b/c"])

    # ---- HIDDEN-REGIME: durable all-or-nothing, resume, parity reconstruct, concurrency ----
    #  (a) all-or-nothing durable upload: if the injected store fails on chunk k, NO chunk must remain
    #      committed (rollback), and the manifest must NOT be finalized (ceph EC commit-all spirit).
    add("hidden_regime", "atomic_upload", "upload failing mid-way rolls back all chunks (no partial)",
        blob=_blob(10, 640), chunk_size=128, fail_on_chunk=3)
    #  (b) resumable download: a download interrupted after k chunks, then resumed from the manifest,
    #      must yield the identical blob and NOT re-download already-fetched chunks (AOF/kafka style).
    add("hidden_regime", "resumable_download", "interrupted download resumes without re-fetch",
        blob=_blob(11, 1000), chunk_size=128, interrupt_after=3)
    #  (c) parity reconstruction: exactly ONE data chunk is missing on download; the candidate must
    #      reconstruct it from the XOR parity chunk and still return the identical blob (minio/EC).
    add("hidden_regime", "parity_reconstruct", "one missing chunk reconstructed from parity",
        blob=_blob(12, 700), chunk_size=128, drop_chunk=2)
    #  (d) too many missing: TWO data chunks missing exceeds single-parity capacity -> download must
    #      FAIL cleanly (raise / None), never fabricate wrong bytes.
    add("hidden_regime", "parity_exceeded", "two missing chunks exceed single parity -> clean failure",
        blob=_blob(13, 700), chunk_size=128, drop_chunks=[1, 3])
    #  (e) concurrent uploads of distinct checkpoints share one store: each blob must download back
    #      identical; no cross-contamination between concurrent transfers.
    add("hidden_regime", "concurrent_transfers", "concurrent distinct transfers do not cross-contaminate",
        n=8, size=384, chunk_size=64)

    return cases


# --------------------------------------------------------------------------------------- graders

def _grade(case, mod):
    axis, kind = case["axis"], case["kind"]
    try:
        # ---------- bad chunk size ----------
        if kind == "bad_chunk_size":
            ct = mod.CheckpointTransfer()
            try:
                ct.pack(case["blob"], chunk_size=case["chunk_size"])
                return False, "bad_chunk_size_not_rejected"
            except Exception:
                return True, "bad_chunk_size_raised"

        # ---------- path escape ----------
        if kind == "path_escape":
            ct = mod.CheckpointTransfer()
            got = ct.normalize_path(case["path"])
            if got != case["expect"]:
                return False, f"path_escape got={got!r} exp={case['expect']!r}"
            return True, "path_escape_ok"

        # ---------- normalize idempotent / batch ----------
        if kind == "normalize_idempotent":
            ct = mod.CheckpointTransfer()
            n1 = ct.normalize_path(case["path"])
            n2 = ct.normalize_path(n1) if n1 is not None else None
            if n1 != n2:
                return False, f"not idempotent {n1!r} != {n2!r}"
            if n1 != _ref_normalize(case["path"]):
                return False, f"normalize wrong {n1!r} exp {_ref_normalize(case['path'])!r}"
            return True, "normalize_idempotent_ok"

        if kind == "normalize_paths":
            ct = mod.CheckpointTransfer()
            for p in case["paths"]:
                if ct.normalize_path(p) != _ref_normalize(p):
                    return False, f"normalize[{p!r}] got={ct.normalize_path(p)!r} exp={_ref_normalize(p)!r}"
            return True, "normalize_paths_ok"

        # ---------- manifest correctness ----------
        if kind == "manifest":
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            ref = _ref_chunks(case["blob"], case["chunk_size"])
            if man.get("num_chunks") != len(ref):
                return False, f"num_chunks {man.get('num_chunks')} != {len(ref)}"
            if man.get("total_size") != len(case["blob"]):
                return False, f"total_size {man.get('total_size')} != {len(case['blob'])}"
            if man.get("blob_crc") != _crc32(case["blob"]):
                return False, "blob_crc mismatch"
            # each chunk crc in manifest must match the reference chunk crc
            chunk_crcs = man.get("chunk_crcs")
            if not chunk_crcs or len(chunk_crcs) != len(ref):
                return False, "chunk_crcs missing/wrong length"
            for i, c in enumerate(ref):
                if chunk_crcs[i] != _crc32(c):
                    return False, f"chunk_crc[{i}] mismatch"
            return True, "manifest_ok"

        # ---------- deterministic pack ----------
        if kind == "deterministic_pack":
            ct = mod.CheckpointTransfer()
            m1 = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            m2 = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            if m1.get("num_chunks") != m2.get("num_chunks") or m1.get("blob_crc") != m2.get("blob_crc") \
                    or m1.get("chunk_crcs") != m2.get("chunk_crcs"):
                return False, "pack not deterministic"
            return True, "deterministic_pack_ok"

        # ---------- basic round-trip through an in-memory store ----------
        if kind == "roundtrip":
            store = _Store()
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            ok = ct.upload(man, store)
            if not ok:
                return False, "upload returned falsy"
            out = ct.download(man, store)
            if out != case["blob"]:
                return False, f"roundtrip mismatch len_out={_ln(out)} len_exp={len(case['blob'])}"
            return True, "roundtrip_ok"

        # ---------- tamper detection ----------
        if kind == "tamper_detect":
            store = _Store()
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            ct.upload(man, store)
            store.tamper(case["tamper_chunk"])  # flip bytes in one stored data chunk
            if case.get("tamper_parity"):
                store.tamper_parity()            # also corrupt parity -> genuinely unrecoverable
            try:
                out = ct.download(man, store)
                if out == case["blob"]:
                    return False, "tampered chunk not detected (returned original?!)"
                if out is None:
                    return True, "tamper_detected_none"
                return False, "tamper produced wrong bytes silently"
            except Exception:
                return True, "tamper_detected_raise"

        # ---------- atomic upload (all-or-nothing) ----------
        if kind == "atomic_upload":
            store = _Store(fail_on_put=case["fail_on_chunk"])
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            try:
                ok = ct.upload(man, store)
            except Exception:
                ok = False
            if ok:
                return False, "upload reported success despite store failure"
            # rollback: NO chunk may remain committed after a failed upload
            if store.num_committed() != 0:
                return False, f"partial commit left {store.num_committed()} chunks (no rollback)"
            return True, "atomic_upload_ok"

        # ---------- resumable download ----------
        if kind == "resumable_download":
            store = _Store()
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            ct.upload(man, store)
            sink = _DownSink(interrupt_after=case["interrupt_after"])
            try:
                ct.download(man, store, sink=sink)
            except _Interrupted:
                pass
            got_before = sink.n_fetched()
            if got_before < 1:
                return False, "no chunks fetched before interrupt"
            sink2 = sink.resume()
            out = ct.download(man, store, sink=sink2, resume_from=sink.have())
            if out != case["blob"]:
                return False, "resumed download mismatch"
            if sink2.refetched_existing():
                return False, "resume re-fetched already-present chunks"
            return True, f"resumable_download_ok total={sink2.n_have()}"

        # ---------- parity reconstruct (one missing) ----------
        if kind == "parity_reconstruct":
            store = _Store()
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            ct.upload(man, store)
            store.drop(case["drop_chunk"])  # a data chunk disappears
            out = ct.download(man, store)
            if out != case["blob"]:
                return False, "parity reconstruction failed to recover blob"
            return True, "parity_reconstruct_ok"

        # ---------- parity exceeded (two missing -> clean failure) ----------
        if kind == "parity_exceeded":
            store = _Store()
            ct = mod.CheckpointTransfer()
            man = ct.pack(case["blob"], chunk_size=case["chunk_size"])
            ct.upload(man, store)
            for d in case["drop_chunks"]:
                store.drop(d)
            try:
                out = ct.download(man, store)
                if out == case["blob"]:
                    return False, "two missing chunks 'recovered' (impossible with single parity)"
                if out is None:
                    return True, "parity_exceeded_none"
                return False, "parity_exceeded produced wrong bytes silently"
            except Exception:
                return True, "parity_exceeded_raise"

        # ---------- concurrent transfers ----------
        if kind == "concurrent_transfers":
            store = _Store()
            ct = mod.CheckpointTransfer()
            blobs = {f"ck{i}": _blob(1000 + i, case["size"]) for i in range(case["n"])}
            mans = {}
            results = {}
            rlk = threading.Lock()

            def do(name):
                man = ct.pack(blobs[name], chunk_size=case["chunk_size"], namespace=name)
                with rlk:
                    mans[name] = man
                ct.upload(man, store)
                out = ct.download(man, store)
                with rlk:
                    results[name] = out

            ths = [threading.Thread(target=do, args=(n,)) for n in blobs]
            for t in ths:
                t.start()
            for t in ths:
                t.join(timeout=30)
            if any(t.is_alive() for t in ths):
                return False, "concurrent_transfers hung"
            for name, b in blobs.items():
                if results.get(name) != b:
                    return False, f"cross-contamination on {name}"
            return True, f"concurrent_transfers_ok n={len(results)}"

    except Exception as e:
        return False, f"exc:{type(e).__name__}:{e}"
    return False, f"unknown_kind:{kind}"


# ---- helpers ----------------------------------------------------------------------------------

def _ln(v):
    return len(v) if isinstance(v, (bytes, bytearray)) else None


class _Interrupted(Exception):
    pass


class _Store:
    """Harness-owned chunk store the candidate uploads to / downloads from. Keys are arbitrary chunk
    ids the candidate chooses (must be stable across pack/upload/download for the same manifest).
    Supports injected put-failure (for atomic-upload), tamper, and drop (for integrity/parity)."""

    def __init__(self, fail_on_put=None):
        self._data = {}
        self._fail_on_put = fail_on_put
        self._put_count = 0
        self._lock = threading.Lock()

    def put(self, key, blob):
        with self._lock:
            if self._fail_on_put is not None and self._put_count >= self._fail_on_put:
                self._put_count += 1
                raise IOError(f"injected store failure at put #{self._put_count}")
            self._put_count += 1
            self._data[key] = bytes(blob)
            return True

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)
            return True

    def exists(self, key):
        with self._lock:
            return key in self._data

    # -- harness-only manipulators (candidate never calls these) --
    def num_committed(self):
        with self._lock:
            return len(self._data)

    def tamper(self, ordinal):
        with self._lock:
            keys = sorted(self._data.keys())
            data_keys = [k for k in keys if "parity" not in k.lower()] or keys
            k = data_keys[ordinal % len(data_keys)]
            b = bytearray(self._data[k]) or bytearray(b"\x00")
            b[0] ^= 0xFF
            self._data[k] = bytes(b)

    def tamper_parity(self):
        with self._lock:
            pkeys = [k for k in self._data if "parity" in k.lower()]
            for k in pkeys:
                b = bytearray(self._data[k]) or bytearray(b"\x00")
                b[0] ^= 0xFF
                self._data[k] = bytes(b)

    def drop(self, ordinal):
        with self._lock:
            keys = sorted(self._data.keys())
            # drop the ordinal-th DATA chunk (data chunks sort before the parity key by convention:
            # candidates should name data chunks so ordinal maps to logical chunk index; we drop by
            # matching a substring 'data' if present, else by position).
            data_keys = [k for k in keys if "parity" not in k.lower()]
            target = data_keys[ordinal % len(data_keys)] if data_keys else keys[ordinal % len(keys)]
            self._data.pop(target, None)


class _DownSink:
    """Sink the candidate's download() writes recovered chunks into; interruptible + resumable."""

    def __init__(self, interrupt_after=None, _carry=None):
        self.interrupt_after = interrupt_after
        self._chunks = dict(_carry) if _carry else {}
        self._prior = set(_carry.keys()) if _carry else set()
        self._order = []
        self._refetched = []

    def write_chunk(self, index, data):
        if index in self._prior:
            self._refetched.append(index)
        self._chunks[index] = data
        self._order.append(index)
        if self.interrupt_after is not None and len([i for i in self._order if i not in self._prior]) \
                >= self.interrupt_after and not self._prior:
            raise _Interrupted()
        return True

    def have(self):
        return sorted(self._chunks.keys())

    def n_fetched(self):
        return len([i for i in self._order if i not in self._prior])

    def n_have(self):
        return len(self._chunks)

    def resume(self):
        return _DownSink(interrupt_after=None, _carry=self._chunks)

    def refetched_existing(self):
        return len(self._refetched) > 0


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
