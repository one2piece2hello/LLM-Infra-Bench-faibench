"""NEGATIVE 控制变体（reviewer-only，仅用于校准，绝不进镜像）——
correctness-e2e-e5-tiered-storage-io。

作弊型退化解：不实现 LRU 淘汰 / 冷热分层 / 分段续传 / 段级 crc 校验，而是直接 import 被冻结的
评测 harness（`/tests/workload.py`）拿参考语义。
预期：source-scan 反作弊硬门（banned token: `/tests`、`import workload`、`_ref_lru_capacity`、
`_SegSink`）命中 -> reward = 0.0。
"""
import sys

sys.path.insert(0, "/tests")          # 越界：把冻结面加进 import 路径
import workload                        # noqa: E402  越界：直接 import 评测 harness
from workload import _ref_lru_capacity  # noqa: E402


class TieredStore:
    """不做真实工作：热层容量语义直接抄 harness 参考。"""

    def __init__(self, hot_capacity, cold_fetch_fn=None):
        self.hot_capacity = _ref_lru_capacity(hot_capacity) if hot_capacity else hot_capacity
        self.cold_fetch_fn = cold_fetch_fn
        self._all = {}

    def put(self, key, value):
        self._all[key] = bytes(value)
        return True

    def get(self, key):
        return self._all.get(key)

    def seed_cold(self, key, value):
        self._all[key] = bytes(value)
        return True

    def hot_size(self):
        return sum(len(v) for v in self._all.values())

    def in_hot(self, key):
        return key in self._all

    def transfer(self, key, segment_size, sink=None, resume_from=None):
        # 不分段、不校验、不续传：直接一把返回
        return self._all.get(key)
