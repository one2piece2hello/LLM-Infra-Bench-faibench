# solution/ — reviewer-only 参考实现

本目录里的补丁**不进镜像**(已被上下文根的 `.dockerignore` 排除)、**判分不运行**,
只在两种场合用:

1. **重标定锚点**(换硬件后):见 `../tests/ref_speedup.caveat.md`
   ```
   -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch
   ```
2. **自证挖洞树的正确性**:`git apply --check -p1 oracle.patch` 必须能干净打在
   `../environment/repo/` 上(正向可、反向不可),这证明该树就是被计分的起始点。

`negative.patch` 是已知坏例(必须得 0),`baseline2.patch` 是中间基线(若有)。
