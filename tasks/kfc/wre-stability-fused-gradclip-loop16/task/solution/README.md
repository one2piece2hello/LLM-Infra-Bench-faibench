# solution/ — 参考实现(不参与判分)

本目录里的文件**不进镜像**(已被上下文根的 `.dockerignore` 排除,且 Dockerfile 的
ANTI-LEAK 断言会在 build 期检查 `kernel_oracle.py` 之类没有被烤进任何一层)、
**判分不运行**,只在两种场合用:

1. **重标定锚点**(换硬件/换解释器后):见 `../tests/ref_speedup.caveat.md`
   ```
   -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py
   ```
2. **自证判分有效**:`negative` 必须得 0、`noop` 应 ≈1.0 被门到 0。

本题是**单文件提交**形制,所以参考实现是整份文件的变体而不是补丁:

| 文件 | 用途 |
|---|---|
| `kernel_negative.py` | 已知坏例,**必须得 0** |
| `kernel_oracle.py` | oracle 参考实现,用于标定锚点(打平锚点 = reward 0.5) |
