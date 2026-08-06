#!/usr/bin/env bash
# solution/solve.sh — kfc/wre-verl-grpo-advantage-loop16
# 把本题的参考实现落成「与 agent 提交等价」的形态,让 oracle 验证从手工传环境变量
# 变成一条命令。本文件不进镜像(没有任何 COPY 取 solution/),判分器也不会运行它。
#
# oracle 形态 : 单文件变体(file),但 🔴 产物不在 solution/ 里 —— solution/ 只有 README.md。
#               tests/test.sh 直接从 tests/ 取 reviewer-only 变体做模式分派:
#                 ORACLE="$TESTS_DIR/oracle_advantage.py"    (--oracle)
#                 NEG="$TESTS_DIR/negative_advantage.py"     (--negative)
#                 NAIVE="$TESTS_DIR/naive_advantage.py"      (--baseline2)
#                 STUB="$TESTS_DIR/stub_advantage.py"        (verifier 的 noop 模式)
#               本脚本只**读**这些文件,绝不改写 tests/ 下的任何字节(规范 H1)。
# 被计分单元  : /app/workspace/submission/advantage_estimators.py
#               来源:tests/test.sh 的 WORK="${WRE_WORKSPACE:-/app/workspace}"; SUB="$WORK/submission/advantage_estimators.py"
#               🔴 本题没有 /app/repo 侧的 git-diff scope 闸门(test.sh 自己写明
#               「The scored unit is submission/advantage_estimators.py (no repo
#               tree / git-diff gate)」),而且 /app/workspace **不在** git 树里
#               (镜像只在 /app/repo 里 git init),所以没有 checkout 复位这一步;
#               整文件覆盖本身就幂等。
# 落地方式    : cp <tests/ 变体> /app/workspace/submission/advantage_estimators.py
#
# 🔴 为什么绝不 commit(规范 H3):本题的被计分单元不走 git-diff 闸门,但镜像仍然自证
#    /app/repo 的 HEAD 恰好只有 1 个 commit(挖洞基线),而 loop16 的 submit.sh --finalize
#    也依赖「HEAD stays at the baked baseline commit」。所以落地一律只改工作树/工作区文件;
#    本脚本落地后会断言 /app/repo 的 HEAD 与 commit 数没变。
#
# ── 落地后判分的那条完整命令(同一个容器里跑完 3 步)────────────────────────
#   PKG=<fai_bench>/tasks/kfc/wre-verl-grpo-advantage-loop16/task
#   IMG=fai/kfc-wre-verl-grpo-advantage-loop16:oss
#   # 1) 起容器。判分容器始终 --network=none;tests/ 和 solution/ 都**没有**烤进镜像
#   #    (Dockerfile 里没有一条 COPY 取它们),所以这两个目录必须挂进来。
#   #    该题 task.toml 声明 gpus=0;锚点的标定硬件/运行栈见 tests/ref_speedup.caveat.md
#   docker run -d --name kfc-solve --network=none \
#     -v "$PKG/tests":/tests -v "$PKG/solution":/solution "$IMG"
#   # 2) 落地参考实现到工作树(本脚本;不 commit)
#   docker exec kfc-solve bash /solution/solve.sh
#   # 3) 判分。🔴 本题的模式选择器是位置参数 $1(或 WRE_MODE 环境变量),
#   #    **不是** KERNELBENCH_VERIFY_MODE(tests/test.sh: MODE="${1:-${WRE_MODE:-candidate}}")。
#   #    必须用 candidate —— 这是 agent 的真实评分路径;oracle/negative 会让
#   #    test.sh 直接改判 tests/ 里的变体,绕过本脚本落地的 workspace 文件。
#   docker exec kfc-solve bash /tests/test.sh candidate
#   docker exec kfc-solve cat /logs/verifier/reward.json   # 判分结果
#   docker rm -f kfc-solve
#
#   期望:oracle → 1.0(实现类二值题:每个可见 case 都过且无作弊);
#         --negative → 0;--baseline2(naive)按 tests/verify_core.py 的判据;
#         --noop(纯净容器)→ 0(workspace 里是 stub,抛 NotImplementedError)。
#   本题带 loop16 harness:`bash /opt/loop/submit.sh` 判的是同一个被计分单元、
#   同样走 candidate 模式(口径一致),所以落地后直接自评也成立。
# ──────────────────────────────────────────────────────────────────────────
#
# 用法(规范 2 的统一 CLI):
#   bash solution/solve.sh              # 落地 oracle(参考实现)
#   bash solution/solve.sh --negative   # 落地 negative(已知坏例,必须得 0)
#   bash solution/solve.sh --baseline2  # 落地 baseline2(naive 中间基线)
#   bash solution/solve.sh --noop       # 把可编辑面复位到烤入基线(对照组)
#
# --noop 语义:把可编辑面复位到烤入基线,使对照组能量到真正的 no-op 值。本题的判分件
#   不在 git 树里,所以复位不是 `git checkout` 而是从题包 environment/workspace/ 的
#   **出厂副本**拷回(等价来源:tests/stub_advantage.py,与出厂副本 sha256 一致);
#   /app/repo 的 HEAD 照旧不动。取不到出厂副本就非 0 退出,绝不拿别的东西冒充基线。
#
# 幂等:被计分单元是整文件覆盖(workspace 不在 git 树里,无需也无法 checkout 复位),
#       重复执行结果一致。
# 失败一律非 0 退出并打印原因(tests/ 变体缺失 / 被计分路径不存在 / 出厂副本取不到 / HEAD 被移动)。
set -uo pipefail

TASK_ID="wre-verl-grpo-advantage-loop16"
WORK="${WRE_WORKSPACE:-/app/workspace}"
SUB="$WORK/submission/advantage_estimators.py"
REPO="${KFC_REPO_DIR:-/app/repo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ printf '[solve.sh %s] %s\n' "$TASK_ID" "$*"; }
die(){ printf '[solve.sh %s] ERROR: %s\n' "$TASK_ID" "$*" >&2; exit 1; }

usage(){ cat <<'USAGE'
usage: bash solve.sh [--oracle | --negative | --baseline2 | --noop]
  (无参数)      落地 oracle   (tests/oracle_advantage.py)
  --negative    落地 negative (tests/negative_advantage.py)
  --baseline2   落地 baseline2(tests/naive_advantage.py —— test.sh 的 baseline2 = NAIVE)
  --noop        复位判分件到出厂基线(从 environment/workspace/ 的出厂副本拷回)
USAGE
}

WHICH=oracle
case "${1:-}" in
  ""|--oracle) WHICH=oracle;   ART=oracle_advantage.py ;;
  --negative)  WHICH=negative; ART=negative_advantage.py ;;
  --baseline2) WHICH=baseline2; ART=naive_advantage.py ;;
  --noop)      WHICH=noop ;;
  -h|--help)   usage; exit 0 ;;
  *)           usage >&2; die "未知参数 '$1'" ;;
esac

# 🔴 本题的参考实现不在 solution/ 里(solution/ 只有 README.md),而是 reviewer-only 地
#    放在 tests/ 下(tests/test.sh: ORACLE="$TESTS_DIR/oracle_advantage.py")。
#    本脚本只**读**它们,绝不改写 tests/ 下的任何字节(规范 H1)。
tests_artifact(){
  local n="$1" c
  for c in "/tests/$n" "$HERE/../tests/$n" "${KFC_TESTS_DIR:-/nonexistent}/$n"; do
    [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}

[ -d "$(dirname "$SUB")" ] || die "被计分目录不存在:$(dirname "$SUB")(WRE_WORKSPACE 不对?)"
[ -f "$SUB" ] || die "被计分单元不存在:$SUB"

# /app/workspace 不在任何 git 树里(镜像只在 /app/repo 里 git init),所以这里没有
# checkout 复位这一步 —— 整文件覆盖本身就是幂等的。/app/repo 的 HEAD 仍然要保持不动。
HEAD_BEFORE=""; N_BEFORE=""
if git -c safe.directory='*' -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  HEAD_BEFORE="$(git -c safe.directory='*' -C "$REPO" rev-parse HEAD)"
  N_BEFORE="$(git -c safe.directory='*' -C "$REPO" rev-list --count HEAD)"
fi

if [ "$WHICH" = noop ]; then
  # --noop 把可编辑面复位到烤入基线,使对照组能量到真正的 no-op 值。
  # 🔴 本题的判分件不在 git 树里,没法 `git checkout` 复位,所以从题包里的**出厂副本**拷回:
  #      environment/workspace/submission/advantage_estimators.py
  #    (Dockerfile: `COPY environment/workspace/ /app/workspace/` —— 它就是镜像里的初始 stub)
  #    tests/stub_advantage.py 与该出厂副本 sha256 完全一致(建包时实测),而 /tests 判分时必挂,
  #    所以把它作为等价来源排在第二位。两条都取不到就非 0 退出,绝不拿别的东西冒充基线。
  SEED=""
  for c in "$HERE/../environment/workspace/submission/advantage_estimators.py" \
           "${KFC_WORKSPACE_SEED:-/nonexistent}/submission/advantage_estimators.py" \
           "/tests/stub_advantage.py" "$HERE/../tests/stub_advantage.py"; do
    [ -f "$c" ] && { SEED="$c"; break; }
  done
  [ -n "$SEED" ] || die "取不到出厂副本 —— 请把题包的 environment/workspace 挂进容器(或设 KFC_WORKSPACE_SEED),\
或至少把 <pkg>/tests 挂到 /tests(tests/stub_advantage.py 与出厂副本 sha256 一致)"
  if cmp -s "$SEED" "$SUB"; then
    say "--noop:判分件本来就是出厂基线,无需复位"
  else
    cp -f "$SEED" "$SUB" || die "复位失败:$SUB"
    say "--noop:已用出厂副本 $SEED 复位 $SUB(原先与基线不同)"
  fi
  if [ -n "$HEAD_BEFORE" ]; then
    [ "$HEAD_BEFORE" = "$(git -c safe.directory='*' -C "$REPO" rev-parse HEAD)" ] \
      || die "$REPO 的 HEAD 被移动了 —— 规范 H3 要求它停在挖洞基线上"
    [ "$N_BEFORE" = "$(git -c safe.directory='*' -C "$REPO" rev-list --count HEAD)" ] \
      || die "$REPO 的 commit 数变了"
  fi
  say "--noop 完成:$SUB = 出厂基线 stub"
  exit 0
fi

SRC="$(tests_artifact "$ART")" || die "找不到 $ART —— 它在题包的 tests/ 里,判分时挂到 /tests。请把 <pkg>/tests 挂进容器(或设 KFC_TESTS_DIR)"
cp -f "$SRC" "$SUB" || die "写入失败:$SUB"

if [ -n "$HEAD_BEFORE" ]; then
  [ "$HEAD_BEFORE" = "$(git -c safe.directory='*' -C "$REPO" rev-parse HEAD)" ] \
    || die "$REPO 的 HEAD 被移动了 —— 规范 H3 要求它停在挖洞基线上"
  [ "$N_BEFORE" = "$(git -c safe.directory='*' -C "$REPO" rev-list --count HEAD)" ] \
    || die "$REPO 的 commit 数变了"
fi

say "已落地 $WHICH:$(basename "$SRC") -> $SUB"
exit 0
