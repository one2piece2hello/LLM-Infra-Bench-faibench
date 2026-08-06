"""NEGATIVE 控制变体（reviewer-only，仅用于校准，绝不进镜像）——
correctness-e2e-e5-checkpoint-transfer-integrity。

这是一个**作弊型退化解**：它不去实现 chunk crc / XOR parity / 全有或全无上传 / 可续传下载，
而是试图直接读取被冻结的评测 harness（`/tests/workload.py`）里的内部参考实现来伪造答案。
预期：`tests/test.sh` 的 source-scan 反作弊硬门（banned token: `/tests`、`import workload`、
`_ref_chunks` …）命中 -> `hard_fails=["banned_path_or_import"]` -> reward = 0.0。

用途：证明 real-work / anti-cheat 探针真的会把「不做真活、直接偷答案」的提交打成 0，
而不是让它拿到高分。
"""
import sys
import zlib

sys.path.insert(0, "/tests")          # 越界：把冻结面加进 import 路径
import workload                        # noqa: E402  越界：直接 import 评测 harness
from workload import _ref_chunks, _ref_parity, _ref_normalize  # noqa: E402


def _crc32(b):
    return zlib.crc32(b) & 0xFFFFFFFF


class CheckpointTransfer:
    """不做真实工作：所有语义都从 harness 的参考实现里抄。"""

    def pack(self, blob, chunk_size, namespace=""):
        chunks = _ref_chunks(blob, chunk_size)
        return {"num_chunks": len(chunks), "total_size": len(blob), "chunk_size": chunk_size,
                "namespace": namespace, "_blob": bytes(blob),
                "_parity": _ref_parity(chunks), "chunk_crcs": [_crc32(c) for c in chunks],
                "blob_crc": _crc32(blob)}

    def upload(self, manifest, store):
        blob = manifest.get("_blob", b"")
        cs = manifest["chunk_size"]
        ns = manifest.get("namespace", "")
        for i in range(manifest["num_chunks"]):
            store.put(f"{ns}data{i:06d}", blob[i * cs:(i + 1) * cs])
        store.put(f"{ns}parity", manifest["_parity"])
        return True

    def download(self, manifest, store, sink=None, resume_from=None):
        # 直接返回 pack 时缓存的原始字节：完全跳过取块、校验、重建
        return manifest.get("_blob")

    def normalize_path(self, path):
        return _ref_normalize(path)
