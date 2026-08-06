<div align="center">

<h1>Φ-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?</h1>

**85 道开源 LLM 基础设施工程题** —— 构建一份公共 Docker 镜像,离线作答,再用随包发布的判分器打分。

### 🔗 [llminfrabench.com](http://llminfrabench.com/)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
&nbsp;![tasks](https://img.shields.io/badge/tasks-85-brightgreen.svg)
&nbsp;[![website](https://img.shields.io/badge/website-llminfrabench.com-8A2BE2.svg)](http://llminfrabench.com/)

###### 🌐&nbsp; [English](README.md) &nbsp;·&nbsp; **简体中文**

</div>

---

**85 道 LLM-infra 工程题**,全部已开源化:每题自带一份**自包含的公共 Dockerfile**,`git clone` + `docker build` 即可复现环境,判分面随包发布。

| 子集 | 题数 | 题型 |
|---|---|---|
| `tasks/kfc/` | 55 | 挖洞重实现:给定一份**功能正确但慢**的实现,只许改声明的 scope 文件,把它做快 |
| `tasks/lh/` | 20 | 同上,Long-horizon 型(多为上游大库的 kernel / 协议层) |
| `tasks/e2e/` | 10 | Type-3 端到端:整个工作树可改,只有一小撮 sha256 冻结的评分面 |

全部 `allow_internet = false`:**作答与判分均离线**,所以环境依赖(含模型权重与数据集)都在 `docker build` 阶段落进镜像。

机器可读索引:`tasks_index.json`(85 条,含每题的包根 / 布局 / GPU / scope / 锚点 / oracle 可得性)。计分公式:`SCORING.md`。

## 目录形制

```
fai_bench/
├── tasks_index.json            85 题索引
├── SCORING.md                  两类 reward 的公式
├── scripts/verify_package.py   包结构 + Dockerfile 可解析性 + 挖洞树自证 自检
└── tasks/
    ├── kfc/<dir>/task/  ┐
    ├── e2e/<dir>/task/  ├ 三个子集的**包根**位置不同,见 tasks_index.json 的 package_root
    └── lh/<dir>/        ┘  (lh 是平铺,kfc/e2e 多一层 task/)
```

**包根**(即 `docker build` 的上下文)下固定只有这些内容:

```
<package_root>/
├── instruction.md        模型唯一可见的输入
├── task.toml             资源、判分入口、primary_metric、docker_image
├── .dockerignore         必须在上下文根(docker 不读 environment/.dockerignore)
├── environment/
│   ├── Dockerfile        ★ 自包含公共配方(公共 base + 公网源 + 版本写死)
│   ├── repo/             挖洞后的工作树(vendored,75 题有)
│   ├── runtime/          entrypoint.sh / timer.sh / run_dev_bench.sh
│   ├── loop/             会话内自评 harness(76 题有;其中 26 题真跑 1~16 轮,
│   │                     其余 50 题是 kfc,MIN=MAX=1 单次提交 —— 见 SCORING.md)
│   └── …                 submission / dev_bench / stubs / workspace(按题)
├── tests/                判分面:test.sh + compute_reward.py + 工作负载 + 锚点
└── solution/             ★ reviewer-only:参考实现 / oracle 补丁(83 题有)
```

## 怎么跑一道题

```bash
cd <package_root>
docker buildx build -f environment/Dockerfile -t <task.toml 里的 docker_image> .
docker run --rm [--gpus all] -it <image>                                     # agent 在容器内作答
docker run --rm [--gpus all] -v "$PWD/tests:/tests:ro" <image> bash /tests/test.sh
cat /logs/verifier/reward.json        # reward 字段即该题得分
```

每题 Dockerfile 头部都写全了 **build / run / score / 重标定** 四条可复制命令,以及该题的版本锚点与工作树来源(`PROVENANCE` 段:从哪个镜像 digest 恢复、用什么方法、怎么验证)。

**构建期需要的网络出口**(按题不同,`tasks_index.json` 可对照):全部题需要 apt 与 PyPI;5 题需要 `git clone` 公共 GitHub;5 道 e2e 题需要从 HuggingFace 拉模型权重(~GB)。**必须用 BuildKit**:题包用了 `RUN <cmd> <<'PY' … PY` heredoc,请用 Docker ≥ 23 + buildx(`docker buildx build`),经典 builder 解析不了这种语法。

> 镜像源环境提示:11 题写的是 `pip install --index-url https://download.pytorch.org/whl/...`。`--index-url` 会**替换**主索引,所以如果你的网络只能走内部 PyPI 镜像,这 11 题会因为拿不到 torch 而失败 —— 把它改成 `--extra-index-url` 即可两种网络都兼容(有公网时行为不变)。其余 45 题本来就用的是 `--extra-index-url`,不受影响。

## 用 runner 一键跑(build → agent → 判分 → reward)

上面那套手动命令,`scripts/run_task.py` 全给你串好了。它**单文件、只用标准库**(Python ≥ 3.8 都能跑;3.11 以下会自动降级到内置的 TOML 解析,不需要 `tomli`),一条命令完成 build → 起容器 → 调 agent → 收产物 → 挂 `tests/` 判分 → 汇总 reward:

```bash
# 用 claude-code / codex 作答一道题(agent CLI 不在镜像里,运行时注入)
python3 scripts/run_task.py --task tasks/kfc/<dir> --agent claude-code --model claude-opus-5 \
    --agent-bin /path/to/claude          # 或 --agent-install 在容器里装
python3 scripts/run_task.py --task tasks/e2e/<dir> --agent codex --model gpt-5.6 --agent-install

# 不调模型的两条自检通路:
python3 scripts/run_task.py --task tasks/lh/<dir> --agent oracle   # 跑 solution/solve.sh,应得该题参考分
python3 scripts/run_task.py --task tasks/kfc/<dir> --agent none    # 判 pristine 基线,应 ≈ 0

# 抽样 / 全量:
python3 scripts/run_task.py --tasks-root tasks --n-tasks 10 --sample-seed 0 --agent claude-code
```

几个要点:

- **提交契约是"工作树式"**,不是 `git commit`。agent 直接编辑允许的文件、把改动**留在工作树里**即可 —— 判分读的是工作树相对烤入基线 commit 的 diff(`pre_artifacts.sh` 用 `git add -AN` + `git diff HEAD` 捕获,不移动 HEAD)。这样任何"会编辑文件"的 scaffold 都能接,换 claude-code / codex / mini-swe-agent 不用改题。**不要让 agent commit** —— HEAD 一动,scope 闸门和取基线的 `git checkout HEAD -- <scope>` 都会失灵,正确解反被判 0。
- **agent CLI 运行时注入**:开源镜像刻意不装任何 agent(已核实 91/91 个 Dockerfile 零命中)。`--agent-bin <宿主机路径>` 把 CLI 只读挂进去,或 `--agent-install` 在容器里装公共 npm 包。gpt 系模型走 `codex`,其余走 `claude-code`。
- **网络**:所有题 `allow_internet = false`。判分**永远** `--network=none`;agent 步默认也离线,只有需要调模型 API 时才按该 agent 的最小白名单开(`--agent-net proxy --net-proxy URL`,由代理强制白名单)。
- **镜像源环境**:build 阶段加 `--build-network host` 让 RUN 步骤能走宿主机可达的 PyPI/apt 镜像(默认关,公网无需)。
- **产出**:`runs/<task>-<agent>-<model>-<ts>/{build.log,agent.log,verify.log,artifacts/,verifier/,run.json}`,以及一行一题的 `runs/summary.jsonl`。

`--agent oracle` 这条通路顺便就是 runner 的自检:它调用每题的 `solution/solve.sh`,把参考实现按"和 agent 提交一样"的方式(工作树、不 commit)落地,再走正常 candidate 判分,应精确命中该题的参考分。

## 两条硬约定

**① `tests/` 随包发布,但绝不烤进镜像。** 判分时才挂到 `/tests`。理由:隐藏用例、强基线、标定锚点都在里面,烤进去等于做题时可读可改。`solution/` 同理 —— 它是 reviewer-only,不进镜像、判分不运行。

**② 性能锚点是标定常量,换硬件必须重标。** reward 形如 `min(1, ln(speedup/ref_speedup)/ln(ref_speedup))`(**打平 ref_speedup 得 0,必须超过**;详见 SCORING.md),`ref_speedup` 判分时只读、不重跑 oracle。77 题有锚点,标定条件写在 `tests/ref_speedup.caveat.md` 或 manifest 的 `hardware_caveat` 字段里,同处给出可复制的重标定命令。**标定环境分两条通道**:GPU 题在 NVIDIA H20,CPU 题在作者的 CPU 通道(Intel Sapphire Rapids)—— 各题写的是它自己的真实标定环境,不要当成统一的一个。

```bash
# 补丁形态(多数 kfc / lh):
docker run --rm [--gpus all] -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
  <image> bash /tests/test.sh
# 单文件形态(部分题的参考实现是整份文件的变体,不是补丁):
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py
```

自证:`noop`(不改动)应 ≈ no-op 值、`negative` 必须得 0。**83 题随包提供参考补丁/oracle**,可直接走这条通路;只有 2 题(`kfc/wro-offload-layer-prefetch-ring-pipeline-loop16`、`kfc/wro-offload-policy-grid-search-loop16`)没有,它们的 caveat 里已注明"锚点换硬件不可比,请自行标定"。

**注意锚点解析链**:`tests/ref_speedup.txt` → 镜像内 `/opt/verifier-correctness-manifest.json` → 1.0。镜像内那份**故意不含真锚点**(它对做题者可读),而 `ref_speedup <= 1` 是 hard gate,所以**必须挂载 `tests/`**,否则会以"锚点无效"大声失败而不是给出错误分数。另外 `tests/ref_speedup.txt` 被 `tr -dc '0-9.'` 解析 —— **不要往里加任何注释**,含数字或小数点的文字会污染锚点值。

## 挖洞树的可复现性(kfc / lh)

这两个子集的起始实现是"上游库被挖掉一块"的树。它**不是 clone 出来的** —— 原始镜像由预置 tarball 组装,没有记录上游 commit,所以 clone 无法钉到被计分的字节。因此 `environment/repo/` 是**从原始镜像恢复并 vendored** 的,并且是**可自证**的:

> `git apply --check -p1 solution/oracle.patch` 必须能干净**正向**打在 `environment/repo/` 上、**反向**打不上 —— 这证明该树正是参考补丁生成时的那个挖洞基线。`scripts/verify_package.py` 会逐题跑这个门。

同家族多题共用一棵上游树时,树里**全家族**的 scope 文件都处于挖洞态(否则一题的镜像会含另一题的答案);build 期有断言检查这一点。

vendored 树是上游代码的原样字节 —— 里面出现的第三方 URL、示例配置、甚至上游自己提交的内部代理提示,都是上游内容,**按自证门的要求必须保持不变**。

## 自检

```bash
python3 scripts/verify_package.py            # 全量自检(85 题;delete_* 归档题自动跳过)
python3 scripts/verify_package.py kfc lh     # 只查某些子集
```

自检是**只读**的,路径全部从脚本自身位置推导,所以 clone 到任何地方都能直接跑。它查的是:
必备件齐全 · `task.toml` 可解析 · `tests/test.sh` 在且性能题的锚点解析得到(不会静默回落 1.0)·
**Dockerfile 可解析**(heredoc 配对、无悬挂续行、`COPY` 源都在上下文里且没被 `.dockerignore` 排除、
每个 `RUN` 的 shell 体过 `bash -n`)· 锚点与 caveat 自洽 · **挖洞树自证**(`oracle.patch` 必须正向
可打、反向打不上)· 无 `__pycache__` / `*.bak` 之类残留 · **可运行三件套**(每题 `pre_artifacts.sh`
与 `solution/solve.sh` 存在、可执行、`bash -n` 过,`solve.sh` 有四态 CLI,`task.toml` 是
`schema_version="2.0"`)。`tasks_index.json` 随包发布,不需要自行重建。

## 许可证

fai_bench **自写的部分**(题目 `instruction.md`/`task.toml`、判分 `tests/**`、参考解 `solution/**`、
loop16 harness `environment/loop*/**`、runner 与自检 `scripts/**`、文档)采用 **Apache License 2.0**
(见 [`LICENSE`](LICENSE))。

`environment/repo/` 下的 **vendored 上游代码**(nanoGPT / torchtitan / vLLM / llama.cpp /
Megatron-LM / ColossalAI / flash-linear-attention 等)以及 build 期从公开源拉取的模型权重与数据集
(Qwen2.5、all-MiniLM-L6-v2、wikitext 等)**各自保留其原始许可与版权**,不在本仓库的 Apache-2.0
授权范围内 —— 详见 [`NOTICE`](NOTICE)。每棵 vendored 树内自带的 `LICENSE`/`COPYING` 文件为其权威许可。
