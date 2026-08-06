# 正确性任务：Checkpoint 分层传输的完整性与恢复（checkpoint transfer integrity）

## 你拿到什么

一个基于 Python 的运行环境。评测器会导入你实现的模块，并驱动其中的一个类：**checkpoint 传输管理器**
`CheckpointTransfer`。这是一道**分层传输 / checkpoint / 完整性**正确性题：你要把一个 checkpoint 二进制块
（`bytes`）切分为固定大小的分片（chunk），为每个分片与整体计算校验和（crc32）并生成 manifest，通过一个
外部注入的分片存储做**全有或全无（all-or-nothing）的持久上传**，在下载时**逐分片校验完整性**、支持从
manifest **断点续传**、并在**恰好一个分片缺失**时用 XOR 奇偶校验分片**重建**它，同时做工件路径的规范化
与冲突检测。语义对齐真实系统：NeMo 的 `S3CheckpointIO`（对象存储保存 + 异步上传 / 本地 staging）、harbor
的 `ArtifactHandler`（manifest 扫描 + 路径规范化 / 冲突 / 分步下载 + 上传）、ceph crimson 的
`ECBackend.submit_transaction`（按 shard 的 erasure-coded subwrite，全部 commit 才完成 durability
future）、minio 的 Reed-Solomon erasure（部分 shard 缺失时重建对象字节流）、redis/valkey 的多段 AOF
manifest（base/incremental/history + 原子切换）。

## 唯一可编辑范围（改动范围之外一律判 0）

你只能修改：

```
/app/submission/ckpt_transfer.py
```

评测器、参考模型、隐藏用例集都在你的工作区之外，**不得读取、复制或修改**，也不得 `import` 评测脚本。

## 必须实现的 API（类名 / 方法名 / 签名需完全一致）

### `class CheckpointTransfer`

- `pack(self, blob, chunk_size, namespace="")`：把 `blob` 切分为固定大小的分片（最后一片可短），返回一个
  **manifest 字典**。至少包含：`num_chunks`（分片数）、`total_size`（blob 字节数）、`chunk_size`、
  `blob_crc`（整块 crc32）、`chunk_crcs`（每片 crc32 组成的列表）、`namespace`。`chunk_size` **必须 > 0**，
  否则抛异常。同时计算**一个 XOR 奇偶校验分片**，使得任意**一个**数据分片缺失时可被重建。
- `upload(self, manifest, store)`：把所有数据分片 + 奇偶校验分片写入 `store`（`store.put(key, bytes)`）。
  **全有或全无**：只要任一 `store.put` 抛异常，就必须把已写入的所有分片**删除**（`store.delete(key)`）并
  返回**假值**；只有整个 checkpoint 持久落盘成功才返回**真值**。
- `download(self, manifest, store, sink=None, resume_from=None)`：返回原始 `blob` 字节，或在**不可恢复**
  时返回 `None`。逐个取数据分片（`store.get(key)`）并**用 manifest 校验其 crc**；**恰好一个**分片缺失 /
  损坏时，必须用奇偶校验分片 + 其余分片**重建**它；**超过一个**缺失时返回 `None`（**绝不能编造字节**）。
  若给出 `sink`，每重建出一个分片就通过 `sink.write_chunk(index, data)` 写出；若给出 `resume_from`（已
  存在的分片索引列表），必须**跳过这些索引，不重复下载**。返回前必须校验拼装出的 `blob_crc`。
- `normalize_path(self, path)`：返回规范化后的工件路径（折叠 `.`、`..`、重复的 `/`、去掉结尾 `/`）；若路径
  **逃逸出根目录**（开头就是无处可退的 `..`）返回 `None`。**必须幂等**（`normalize(normalize(p)) ==
  normalize(p)`）。

### Store 协议（评测器提供，你只**调用**它，不实现）

`put(key, bytes) -> bool`（可能抛异常）、`get(key) -> bytes|None`、`delete(key) -> bool`、
`exists(key) -> bool`。你自己选择**稳定的**分片 key（同一 manifest 的 pack/upload/download 必须一致）。
**约定**：数据分片的 key **不得包含子串 `parity`**，奇偶校验分片的 key **必须包含 `parity`**——评测器在
模拟"丢失一个数据分片"时以此区分数据片与校验片。

## 评分

本题为**正确性（实现类）**任务，无性能指标，得分是**二值**的：

- **全部**评测用例通过，且没有触发任何作弊 / 禁改条件 ⇒ **reward = 1.0**
- 只要**有任意一个用例失败**，或触发任何一条硬失败条件 ⇒ **reward = 0.0**

没有"按通过率给部分分"这回事：`0.99` 的通过率和 `0.0` 的通过率得分相同（都是 0.0）。评测器仍会在
`/logs/verifier/reward.json` 里输出 `tests: {passed, total}` 与分轴明细，但那只是给离线诊断用的，
**不影响得分**。

评测用例沿六个维度分级覆盖：常规（round-trip、manifest 正确）、边界（空 blob、单字节、
blob 小于 chunk_size、超多分片）、退化（全零 blob、重复分片、`chunk_size==1`）、错误（非法 chunk_size、
路径逃逸、篡改分片被检出）、元变换（pack 确定性、路径规范化幂等、批量路径规范化）、隐藏工况（**上传中途
失败全部回滚**、**下载断点续传不重复取**、**一个分片缺失用奇偶校验重建**、**两个分片缺失超出单校验能力
干净失败**、**并发多路传输互不污染**）。触发硬失败（改动范围之外、缺入口、硬编码评测器路径、评测器崩溃）
会让本题直接判 0。


### 提交次数

本题为**单次评测**任务，没有多轮提交循环：不存在最少提交次数，也不存在"必须连续几轮无提升才能
结束"的要求。你认为实现完成即可结束。

## 正确性提示（最容易踩坑的地方）

- **manifest**：`chunk_crcs` 必须逐片与参考一致；`blob_crc` 是整块 crc32；`num_chunks` 与 `total_size` 要准。
- **全有或全无**：`store.put` 抛异常后，已写入的分片必须**全部删除**、返回假值——不得留下半个 checkpoint。
- **完整性**：下载时每片都要**校验 crc**；篡改或缺失的分片当作故障处理。
- **奇偶重建**：恰好一个分片缺失时用 XOR 校验分片重建（注意**最后一片可能短**，重建后要按真实长度截断并
  校验其 crc）；**超过一个**缺失时**返回 `None`**，绝不编造。
- **断点续传**：`resume_from` 给出的索引不重复下载；拼装出的 blob 必须与原块 `blob_crc` 一致。
- **并发**：多个 checkpoint 并发 pack/upload/download 共用一个 store 时，各自的分片 key 要用 `namespace`
  隔离，**互不污染**。

## 规则与红线

- 只改 **可编辑范围** 内的文件；保持上述类名 / 方法名 / 签名。
- 不得读取、复现或修改评测脚本、参考模型或隐藏用例；不得 `import` 评测 harness（如 `workload`）；不得在
  源码里硬编码评测器路径（`/tests`、`/opt/verifier`、`/logs/verifier`、`reward.json` 等）。
- 不得联网，不得抓取上游任何等价实现来"抄答案"（`git clone`/`fetch`/`pull`、加 git remote、
  `pip download`、`wget`/`curl`），也不得绕过环境的网络限制。违反任一条整题判 0。

## 提交方式

把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `git commit`
(判分读工作树;仓库的 HEAD 必须停在初始基线 commit 上)。
