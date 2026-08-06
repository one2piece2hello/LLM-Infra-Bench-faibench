# fai_bench — 计分

**English**: [`SCORING.md`](SCORING.md) · **中文**: `SCORING.zh-CN.md`(本文件)

每题的权威实现是它自己的 `tests/compute_reward.py`;本文档说明两类 reward 的形状,以便读结果时不误判。判分产物固定落在 `/logs/verifier/`:

```
reward.json     结构化结果:reward + 分项(逐用例通过数、配对加速比、诊断量)
reward.txt      与 reward.json 的 reward 同值的纯文本
```

**`reward.json` 的 `reward` 字段就是该题最终得分**,不需要二次换算(不需要像某些 harness 那样把 `score` 当保守值、再由 leaderboard 另算部分分)。

## 两类 reward

| 类别 | 题数 | 值域 | 形状 |
|---|---|---|---|
| 性能类 | 77 | 连续 [0, 1] | 先过正确性门,再按**相对 oracle** 的对数加速比给分 |
| 实现类 | 8 | 二值 {0.0, 1.0} | 全部隐藏用例通过且无门触发才 1.0 |

### 性能类:对数加速比,**oracle 是 0 分起点**

77 道性能题统一用 `reward_md_log_speedup_v2_oracle_zero`:

```
speedup ≤ ref_speedup  ⇒  reward = 0
speedup > ref_speedup  ⇒  reward = min(1.0, ln(speedup / ref_speedup) / ln(ref_speedup))
                                                                        值域 [0, 1]
```

三个锚点决定了它的读法:

- `speedup ≤ ref_speedup`(**没超过出题期标定的 oracle**)⇒ **reward = 0**
- `speedup == ref_speedup^1.5` ⇒ **reward = 0.5**
- `speedup ≥ ref_speedup²` ⇒ **封顶 1.0**

**要点:打平 oracle 得 0 分,必须"超过 oracle"才开始得分。** 这条曲线是旧曲线
`r_v1 = min(1, 0.5·ln(speedup)/ln(ref_speedup))`(打平 oracle 给 0.5)的线性变换:

```
r_v2 = max(0, 2·r_v1 − 1)
```

改动动机是区分度:旧曲线把大量提交挤在 0.5(追平 oracle)附近,真正想区分的"能否超过 oracle、超多少"只剩半个量程。新曲线把整个 [0,1] 让给"超过 oracle 之后"。

**⚠️ 新旧分数不可直接比较。** 历史成绩若需换算,`r_v1 = (r_v2 + 1) / 2`,且仅在 `r_v2 > 0` 时成立
(v1 落在 [0, 0.5] 的那半段在 v2 里全被压成 0,信息不可逆)。**`ref_speedup` 本身没变**,不需要重新标定锚点。

`ref_speedup` 是出题期标定的常数,写死在该题 `tests/` 下的 manifest 里,判分时只读不算 —— 所以同一道题的分数跨模型、跨时间可比。例如 `kv-traffic-sol` 的 `ref_speedup = 2.5799`:speedup 2.58(打平)得 **0**、≈4.14(`ref^1.5`)得 0.5、≥6.656(`ref²`)得 1.0。

**"speedup" 不总是墙钟加速比** —— 它是该题声明的 `perf_metric` 之比,共三种:

| perf_metric | 含义 | 例题 |
|---|---|---|
| 墙钟/带宽加速比 | ABBA 配对(baseline/candidate 交替若干对,每对取各计时用例的几何平均),再跨对取中位数 | `kv-traffic-sol`、`varlen-prefill-attn-sol`、`vllm-scheduler` |
| `quality_at_fixed_budget` | 固定预算下的质量比,如 `baseline_bpb / candidate_val_bpb` | `a3-moe-train-budget`、`a4-token-efficiency-budget` |
| `quality_under_budget` | 固定字节预算下的检索质量比,如 64 B/vector 下 `candidate_nDCG@10 / baseline_nDCG@10` | `embed-compress-golf`(强基线 nDCG@10 = 0.459151,ref = 1.4290) |

**ABBA 配对**是性能题测量的关键手法:baseline 与 candidate 交替测量成对,取每对的比值,再跨对取中位数。这样机器噪声、热身效应、频率漂移对两边同等作用,不会被算成加速。

### 实现类:二值,任何一处不过即 0

8 道实现题的 reward 只有 0.0 和 1.0:

```
reward = 1.0  当且仅当  全部隐藏用例通过  且  无作弊/禁改门触发
reward = 0.0  其它一切情况
```

用例数各题不同,从十几个到上百个隐藏用例/门不等(逐题的实际数量由该题 `tests/` 决定)。**逐用例通过数与分项诊断照样写进 `reward.json`,但绝不把分数移出 {0.0, 1.0}** —— 它们只供离线分析。

有些实现类题**带计时测量**,但计时**不进分数**、只作诊断,或只作为一个"必须跨过强基线"的**前置门**(例如要求若干隐藏 workload 的配对比值中位数 > 1.0 且非退化)。这类题仍是二值的:门全过 = 1.0,否则 0.0 —— **看到 `reward.json` 里有 `speedup` 字段不代表它是性能题**,以 `reward_class` / `reward_formula` 为准。

## 归零门(hard fail)

以下任一命中,该题直接 **reward = 0**,与测量结果无关:

1. **构建/导入/就绪失败** —— 提交的代码起不来
2. **正确性套件任一 case fail** —— 性能题的正确性门是全或无,不给部分分
3. **作弊检测命中** —— 冻结面(frozen surface)被篡改、配对比值恒等(伪造测量)、加速比不合物理(implausible)
4. **触碰 `forbidden_edit_paths`** —— `task.toml` 里列出的路径受 sha256 冻结
5. **性能题 `speedup ≤ 1`** —— 没赢过强基线等于没有改进
6. **`ref_speedup` 缺失或 ≤ 1** —— 锚点不可信时拒绝给分,而不是给一个可疑的分

**区分"归零门"与"曲线取 0"**:`1 < speedup ≤ ref_speedup`(赢了强基线但没超过 oracle)**不是** hard fail ——
它是**曲线本身取 0**,`hard_fail_reasons` 保持为空。`hard_fail` 的语义是"这次运行无效/作弊",不用来表示
"分数低";读结果时若看到 `reward = 0` 且 `hard_fail_reasons` 为空,含义就是"跑通了、但没超过 oracle"。

反作弊不只靠这些门:每题 `tests/test.sh` 在判分前还会做源码扫描(禁止引用 verifier 内部路径如 `/tests/`、`compute_reward`、`reward.json`)、必要时强制从源码重建、以及符号级检查(`ldd`/`nm` 查是否偷链原库)。

## 提交预算:按【子集 × 题型】两维决定

| 子集 | 性能类 | 实现类 |
|---|---|---|
| `kfc`(55 题) | **1 次** | **1 次** |
| `lh`(20 题) | 1~16 次 | 1 次 |
| `e2e`(10 题) | 1~16 次 | 1 次 |

**`kfc` 全子集单次提交,与题型无关** —— 55 道都只有一次评分机会,分两种形态:

- **50 道**装了 loop harness 但 `MIN_SUBMISSIONS=MAX_SUBMISSIONS=1`:第一次 `bash /opt/loop/submit.sh` 评分后**立即自动定稿**,没有第二次带分的尝试(再次调用只会重新定稿同一份已记录的快照,不给新改动评分)
- **5 道**(`chunked-mlp-recompute`、`ckpt-dcp-meta-bbox-merge`、`mamba-zoh-discretize`、`s4-fft-longconv`、`wre-verl-grpo-advantage-loop16`)没有 `submit.sh`:改动留在工作树里,由会话结束后挂载的 `tests/test.sh` 一次性判分

只有 **`lh`/`e2e` 的性能题**——共 **26 道**——跑 1~16 轮协议。上限 16 **不是硬要求**:agent 自行决定何时 `submit.sh --finalize` 收手(k=16 时自动定稿),不必凑满。实现类无论哪个子集都是单次。

每题的预算在**三处**声明且必须一致:`environment/loop/submit.sh` 的 `MIN_SUBMISSIONS`/`MAX_SUBMISSIONS`、`environment/loop/private/manifest.json`、`task.toml` 的 `[loop]` 段。三者不一致即为题包缺陷。

## loop16 题的分数取自"最佳一轮",不是最后一次编辑

上面那 26 道跑 1~16 轮的题,`submit.sh --finalize` 会把 `/logs/loop/best.json` 指向的**历史最佳那一轮**植入为被判分的产物。这意味着:

- agent 最后一次编辑若比中途更差,**不影响得分**
- 每轮的逐轮测量在 `/logs/loop/state.jsonl`,轮次计数在 `/logs/loop/count`
- 若会话被超时掐断,`--finalize` 仍会植入当时最好的一轮 —— 因此**一个非零 reward 不能证明会话正常收尾**。要判断是否完整作答,看会话是否正常结束,而不是看 reward 是否 > 0

**新曲线下的一个已知行为**:`best.json` 是按 dev reward 严格递增更新的。当某次会话**全程都没超过 oracle** 时,
每一轮的 dev reward 都是 0,于是"最佳轮"会停在第 1 轮 —— `best_so_far` 反馈与最终植入的产物都是最早那棵树,
而不是 speedup 最高那棵。**这不影响分数**(全程低于 oracle,植入哪一轮最终都是 0),但要注意这种情况下最终植入的产物
是最早那一轮,而非 speedup 最高那一轮。agent 的进步信号不受影响:每轮反馈里的 `dev_speedup` 照常给出,所以它看得见
1.05× → 1.99× 的改进,只是 reward 一直是 0。

## 聚合到 bench 级别

`tasks_index.json` 给出每题的 `category` 与 `medium_topic`/`big_topic`。做模型横比时:

- 性能类与实现类**不要直接混算平均** —— 前者连续、后者二值,混算会让实现类的 1.0 淹没性能类的差异
- **跨版本比较必须确认双方用的是同一条 reward 曲线** —— 看 `reward.json` 的 `schema_version`(v2 曲线是 `kernelbench_reward_v3_oracle_relative`);不同曲线下的分数不可直接比较
