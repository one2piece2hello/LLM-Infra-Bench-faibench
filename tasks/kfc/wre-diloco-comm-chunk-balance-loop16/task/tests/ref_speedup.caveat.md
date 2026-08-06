# ref_speedup 硬件/环境 caveat(开源版新增,不改变判分逻辑)

`ref_speedup.txt` 里的 **1.66973** 是 oracle 标定常量,判分时只读、不重跑 oracle:

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

`tests/test.sh` 的解析顺序是 **`tests/ref_speedup.txt` → `/opt/verifier-correctness-manifest.json` → 1.0**;
回落到 1.0 会让所有 reward 归 0,所以这个文件必须存在且 > 1。

## 标定条件

- 硬件:**the authoring CPU lane (Intel Sapphire Rapids)**
- 运行栈:**Python 3.11 标准库**(本题不用 torch / numpy,判分路径全是纯 Python)
- 线程:`OMP/MKL/OPENBLAS_NUM_THREADS=1`(已写进镜像 ENV)
- 度量:`proxy_bottleneck_bytes` —— 是**确定性计数**而不是墙钟计时,所以这个锚点对硬件的敏感度远低于计时型锚点,
  但仍与解释器版本绑定。

## 硬件/环境不一致时如何重标

verifier 原生支持模式分派,**按文件路径**传入参考实现(从不烤进镜像):

```bash
docker run --rm -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py \
  fai/kfc-wre-diloco-comm-chunk-balance-loop16:oss bash /tests/test.sh
# 从 /logs/verifier/ 读实测 speedup,写回 tests/ref_speedup.txt
```

自证:`noop`(不覆盖 submission)应 ≈1.0 → 被门到 0;
`negative`(`-e KERNELBENCH_VERIFY_MODE=negative -e KERNELBENCH_NEGATIVE_FILE=/patches/kernel_negative.py`)必须 0。

✅ 本题**随包提供**参考实现的单文件变体(`solution/kernel_oracle.py`),所以重标定通路是**可直接执行**的。
