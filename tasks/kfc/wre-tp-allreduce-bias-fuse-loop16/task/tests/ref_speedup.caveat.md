# ref_speedup 硬件/环境 caveat(开源版新增,不改变判分逻辑)

`ref_speedup.txt` 里的 **7.8501** 是 oracle 标定常量,判分时只读、不重跑 oracle:

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

`tests/test.sh` 的解析顺序是 **`tests/ref_speedup.txt` → `/opt/verifier-correctness-manifest.json` → 1.0**;
回落到 1.0 会让所有 reward 归 0,所以这个文件必须存在且 > 1。

## 标定条件

- 硬件:**NVIDIA H20**
- 运行栈:`torch 2.7.0`(原始私有基座声明 torch 2.5.1/triton 3.1.0,任务 overlay 又 `pip install torch==2.7.0`,锚点即在 2.7.0 下标定)
- 线程:`OMP/MKL/OPENBLAS_NUM_THREADS=1`(计时稳定性的一部分,已写进镜像 ENV)

## 硬件不一致时如何重标

verifier 原生支持模式分派,参考实现**按文件路径传入、从不烤进镜像**:

```bash
docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py \
  fai/kfc-wre-tp-allreduce-bias-fuse-loop16:oss bash /tests/test.sh
# 从 /logs/verifier/ 读实测 speedup(建议 >=3 次取中位数),写回 tests/ref_speedup.txt
```

自证:`noop`(不覆盖 submission)应 ≈1.0 → 被门到 0;`negative`(`-e KERNELBENCH_VERIFY_MODE=negative -e KERNELBENCH_NEGATIVE_FILE=/patches/kernel_negative.py`)必须 0。

✅ 本题**随包提供**参考实现的单文件变体(`solution/kernel_oracle.py`),所以重标定通路是**可直接执行**的(verifier 按文件而不是按补丁分派模式)。
