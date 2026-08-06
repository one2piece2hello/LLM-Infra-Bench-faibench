# solution/ — 参考实现(不参与判分)

本目录里的补丁**不进镜像**(已被上下文根的 `.dockerignore` 排除)、**判分不运行**,
只在两种场合用:

1. **重标定锚点**(换硬件后):见 `../tests/ref_speedup.caveat.md`
   ```
   -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch
   ```
2. **自证起始树**:`git apply --check -p1 oracle.patch` 必须能干净打在
   `../environment/repo/` 上(正向可、反向不可)。这证明 `../environment/repo/` 就是
   被计分的、**已被降级的起始实现**,而 oracle 补丁正是把它恢复成参考实现的那一步。

| 文件 | 用途 |
|---|---|
| `baseline2.patch` | 中间基线(正确但朴素),用于交叉核对 |
| `oracle.patch` | oracle 参考实现补丁,用于标定锚点(打平锚点 = reward 0.5) |
