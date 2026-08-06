#!/usr/bin/env python3
"""ORACLE (reviewer-only, NOT baked into the image) for correctness-e2e-e5-checkpoint-transfer-integrity.

A complete, correct implementation of the checkpoint-transfer manager. Verified against the harness's
own reference model; oracle mode must score correctness_frac == 1.0. Applied at score time via
KERNELBENCH_ORACLE_PATCH; the Dockerfile never COPYs it. Oracle patch = diff against the shipped
starter stub environment/workspace/submission/ckpt_transfer.py.

Design notes:
  * pack: split blob into fixed-size data chunks (last may be short); compute per-chunk crc32, blob
    crc32, and a single XOR parity chunk over zero-padded data chunks (minio/EC single-parity variant).
    Manifest is a plain dict: {num_chunks, total_size, chunk_size, blob_crc, chunk_crcs, namespace}.
  * chunk store keys: data chunks -> f"{ns}data{index:06d}", parity -> f"{ns}parity". Data keys sort in
    logical order and never contain the substring 'parity' so the harness drop() maps ordinal->chunk.
  * upload: all-or-nothing. Put every data chunk + parity; on ANY store failure, delete everything
    already committed and return False (ceph EC commit-all-or-rollback spirit). Finalize only on success.
  * download: fetch each data chunk, verify its crc against the manifest; a missing/failed chunk is
    reconstructed from parity + the other chunks IF at most one is missing; more than one missing ->
    fail cleanly (return None). A crc mismatch that cannot be repaired -> None. Supports resume via a
    sink + resume_from (skip already-present chunk indices).
  * normalize_path: collapse '.', '..', duplicate '/', trailing '/'; reject root escapes (return None).
"""
import threading
import zlib


def _crc32(b):
    return zlib.crc32(b) & 0xFFFFFFFF


class CheckpointTransfer:
    def __init__(self):
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------- pack
    def pack(self, blob, chunk_size, namespace=""):
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        chunks = [blob[i:i + chunk_size] for i in range(0, len(blob), chunk_size)] or [b""]
        chunk_crcs = [_crc32(c) for c in chunks]
        parity = bytearray(chunk_size)
        for c in chunks:
            for i in range(len(c)):
                parity[i] ^= c[i]
        return {
            "num_chunks": len(chunks),
            "total_size": len(blob),
            "chunk_size": chunk_size,
            "blob_crc": _crc32(blob),
            "chunk_crcs": chunk_crcs,
            "namespace": namespace,
            "_parity": bytes(parity),   # carried in-manifest; upload writes it to the store
            "_blob": bytes(blob),       # stashed source bytes so upload can derive chunk payloads
        }

    # --------------------------------------------------------------------- store key helpers
    @staticmethod
    def _dkey(ns, i):
        return f"{ns}data{i:06d}"

    @staticmethod
    def _pkey(ns):
        return f"{ns}parity"

    # ------------------------------------------------------------------ upload (all-or-nothing)
    def upload(self, manifest, store):
        ns = manifest.get("namespace", "")
        n = manifest["num_chunks"]
        cs = manifest["chunk_size"]
        parity = manifest.get("_parity")
        committed = []
        try:
            # we need the source bytes: the manifest does not carry data chunks, so upload must be
            # given them. In this contract the candidate re-derives chunks from a stashed blob kept on
            # the manifest at pack time. To stay stateless here we stash the blob under '_blob'.
            blob = manifest.get("_blob")
            if blob is None:
                # reconstruct chunk payloads is impossible without the blob; require pack to stash it.
                raise ValueError("manifest missing _blob for upload")
            for i in range(n):
                seg = blob[i * cs:(i + 1) * cs]
                k = self._dkey(ns, i)
                store.put(k, seg)
                committed.append(k)
            if parity is not None:
                pk = self._pkey(ns)
                store.put(pk, parity)
                committed.append(pk)
        except Exception:
            # rollback everything committed so far -> no partial state
            for k in committed:
                try:
                    store.delete(k)
                except Exception:
                    pass
            return False
        return True

    # ------------------------------------------------------- download (verify + parity + resume)
    def download(self, manifest, store, sink=None, resume_from=None):
        ns = manifest.get("namespace", "")
        n = manifest["num_chunks"]
        cs = manifest["chunk_size"]
        crcs = manifest["chunk_crcs"]
        already = set(resume_from) if resume_from else set()

        recovered = {}   # index -> bytes
        missing = []
        for i in range(n):
            if i in already:
                continue  # resume: skip already-present chunks (do not re-fetch)
            raw = store.get(self._dkey(ns, i))
            if raw is None:
                missing.append(i)
                continue
            if _crc32(raw) != crcs[i]:
                # corrupt: treat as missing so parity can repair a single fault; if >1, we fail below
                missing.append(i)
                continue
            recovered[i] = raw
            if sink is not None:
                sink.write_chunk(i, raw)

        # reconstruct at most ONE missing chunk from parity + the others
        if len(missing) > 1:
            return None  # exceeds single-parity capacity -> clean failure
        if len(missing) == 1:
            par = store.get(self._pkey(ns))
            if par is None:
                return None
            m = missing[0]
            rec = bytearray(par)
            for i in range(n):
                if i == m:
                    continue
                c = recovered.get(i)
                if c is None:
                    # a resumed chunk we skipped: we must have it to reconstruct -> pull from sink carry
                    c = self._from_resume(sink, i)
                    if c is None:
                        return None
                for j in range(len(c)):
                    rec[j] ^= c[j]
            # trim to the true length of chunk m (only the LAST chunk may be short)
            mlen = cs if m < n - 1 else (manifest["total_size"] - m * cs)
            recovered[m] = bytes(rec[:mlen])
            if _crc32(recovered[m]) != crcs[m]:
                return None
            if sink is not None:
                sink.write_chunk(m, recovered[m])

        # assemble (include resumed chunks carried on the sink)
        parts = []
        for i in range(n):
            if i in recovered:
                parts.append(recovered[i])
            elif i in already:
                c = self._from_resume(sink, i)
                if c is None:
                    return None
                parts.append(c)
            else:
                return None
        out = b"".join(parts)
        if _crc32(out) != manifest["blob_crc"]:
            return None
        return out

    @staticmethod
    def _from_resume(sink, i):
        if sink is None:
            return None
        chunks = getattr(sink, "_chunks", None)
        if isinstance(chunks, dict):
            return chunks.get(i)
        return None

    # ------------------------------------------------------------------- path normalization
    def normalize_path(self, path):
        parts = []
        for seg in path.split("/"):
            if seg in ("", "."):
                continue
            if seg == "..":
                if not parts:
                    return None
                parts.pop()
            else:
                parts.append(seg)
        return "/".join(parts)
