# ref_speedup 硬件/环境 caveat(开源版说明,不改变判分逻辑)

**本题不使用性能锚点。**

本题是**实现类**任务:reward 为**二值 0/1**(见 `tests/compute_reward.py` 与
`tests/verify_core.py` 的 `IMPLEMENTATION class -> BINARY`),判分只看正确性门是否全过,
不做 speedup 与 `ref_speedup` 的对数比较。因此:

- 本目录**没有** `ref_speedup.txt`,也不需要有;
- **换硬件不需要重标定** —— 二值判分与主机性能无关;
- `/opt/verifier-correctness-manifest.json` 对本题不承载锚点。

## 参考实现

参考实现随包提供:`tests/oracle_advantage.py`(必须全过正确性门)。
配套的对照件:`tests/naive_advantage.py`(正确但朴素)、`tests/negative_advantage.py`
(已知坏例,必须得 0)、`tests/stub_advantage.py`(起始桩)。

被计分单元是 `workspace/submission/advantage_estimators.py`(单文件提交形制)。
