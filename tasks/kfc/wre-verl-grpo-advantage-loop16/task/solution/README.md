# solution/ — 参考实现(不参与判分)

本题的参考实现**不放在这里**,而是随 `tests/` 一起提供:

| 文件 | 用途 |
|---|---|
| `../tests/oracle_advantage.py` | oracle 参考实现(必须全过正确性门) |
| `../tests/naive_advantage.py` | 正确但朴素的对照实现 |
| `../tests/negative_advantage.py` | 已知坏例,**必须得 0** |
| `../tests/stub_advantage.py` | 起始桩(被计分单元的初始形态) |

本题是实现类、二值判分,**没有性能锚点、不需要重标定** —— 见 `../tests/ref_speedup.caveat.md`。
