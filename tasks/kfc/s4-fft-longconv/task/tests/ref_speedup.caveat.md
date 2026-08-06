# ref_speedup 硬件/环境 caveat(开源版新增,不改变判分逻辑)

本题是**实现类**任务:reward 为二值 0/1(全部正确性用例通过且无作弊 -> 1.0,否则 0.0),**不使用 ref_speedup 锚点**,因此无需重标定。下方公式仅对性能类任务适用:

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

本题不随包提供 `tests/ref_speedup.txt`,判分也不读它 —— 二值 reward 与锚点无关。
`environment/verifier-correctness-manifest.json` 里的 `oracle_ms` 仅作诊断记录,不参与判分。

## 标定条件

- 硬件:**NVIDIA H20**
- 运行栈:`torch 2.7.0`(原始私有基座声明 torch 2.5.1/triton 3.1.0,任务 overlay 又 `pip install torch==2.7.0`,锚点即在 2.7.0 下标定)
- 线程:`OMP/MKL/OPENBLAS_NUM_THREADS=1`(计时稳定性的一部分,已写进镜像 ENV)

## 换硬件时要做什么

**不需要重标定** —— 本题 reward 是二值的,与硬件无关。上面的「标定条件」只是记录
参考实现当初被验证的环境;换卡后正确性用例应当照常全过。

参考实现与反例是**单文件形式**(不是 patch),覆盖到可编辑文件上即可:

- `solution/kernel_oracle.py` —— 参考实现,应当得 1.0
- `solution/kernel_negative.py` —— 已知坏例,**必须**得 0.0

两者都不进镜像(已被 `.dockerignore` 排除),判分时也从不运行。

计时仍会作为诊断元数据记录(`environment/verifier-correctness-manifest.json`
的 `oracle_ms` 是 H20 验证时的记录值),但不参与 reward。
