#!/usr/bin/env bash
# solution/solve.sh — kfc/wro-offload-policy-grid-search-loop16
# 🔴 本题不随包提供参考实现 —— 本脚本只负责把这件事说清楚,并以 exit 2 表明
# 「无法落地」。它不进镜像(没有任何 COPY 取 solution/),判分器也不会运行它。
#
# oracle 形态 : 🔴 无 —— 本题不随包提供参考实现(solution/ 下只有 README.md)。
# $REPO       : /app/repo                      (tests/test.sh: REPO=/app/repo)
# 被计分单元  : /app/repo/offload_policy.py
#               来源:tests/test.sh 的 SCOPE=("offload_policy.py")
# 本脚本行为  : 落地类模式只打印说明并 exit 2;--noop 例外 —— 它照统一语义把
#               可编辑面复位到烤入基线后 exit 0(这一步本题是做得到的)。
#
# ── 落地后判分的那条完整命令(同一个容器里跑完 3 步)────────────────────────
#   PKG=<fai_bench>/tasks/kfc/wro-offload-policy-grid-search-loop16/task
#   IMG=fai/kfc-wro-offload-policy-grid-search-loop16:oss
#   # 1) 起容器。判分容器始终 --network=none;tests/ 和 solution/ 都**没有**烤进镜像
#   #    (Dockerfile 里没有一条 COPY 取它们),所以这两个目录必须挂进来。
#   #    该题 task.toml 声明 gpus=0;锚点的标定硬件/运行栈见 tests/ref_speedup.caveat.md
#   docker run -d --name kfc-solve --network=none \
#     -v "$PKG/tests":/tests -v "$PKG/solution":/solution "$IMG"
#   # 2) 🔴 本题无参考实现可落地 —— solve.sh 会打印说明并 exit 2。这条通路只能用来
#   #    判 agent 的候选树,或(纯净容器里直接跳到第 3 步)判 no-op 基线。
#   # 3) 判分。KERNELBENCH_VERIFY_MODE 必须留在默认的 candidate —— 这就是 agent 的
#   #    真实评分路径;设成 oracle/negative 会让 test.sh 自己再落地一遍,绕过本脚本。
#   docker exec -e KERNELBENCH_VERIFY_MODE=candidate kfc-solve bash /tests/test.sh
#   docker exec kfc-solve cat /logs/verifier/reward.json   # 判分结果
#   docker rm -f kfc-solve
#
#   期望:--noop(纯净容器)→ 0(挖洞基线过不了正确性门);oracle/negative 无从验证。
#   本题带 loop16 harness:`bash /opt/loop/submit.sh` 判的也是同一棵工作树、
#   同样走 candidate 模式(口径一致)。
# ──────────────────────────────────────────────────────────────────────────
#
# 用法: bash solution/solve.sh [--oracle|--negative|--baseline2|--noop]
#   --noop 之外的任何模式都只打印说明并 exit 2(本题没有参考实现产物)。
#
# --noop 语义(与其余 53 题一致):把可编辑面复位到烤入基线(与 verifier 取基线的
#   方式一致 —— `git checkout HEAD -- <scope>`,HEAD 不动),使对照组能量到真正的
#   no-op 值,然后 exit 0。本题唯一缺的是「落地参考实现」这一步。
set -uo pipefail

TASK_ID="wro-offload-policy-grid-search-loop16"
REPO="${KFC_REPO_DIR:-/app/repo}"
SCOPE=("offload_policy.py")

say(){ printf '[solve.sh %s] %s\n' "$TASK_ID" "$*"; }
die(){ printf '[solve.sh %s] ERROR: %s\n' "$TASK_ID" "$*" >&2; exit 1; }
GIT(){ git -c safe.directory='*' -C "$REPO" "$@"; }

# --noop 仍然支持:本题虽然没有参考实现可落地,但「把可编辑面复位到烤入基线」是可满足的,
# 且 runner 需要它来量 no-op 对照值,所以照 D 任务的统一语义实现并 exit 0。
if [ "${1:-}" = "--noop" ]; then
  GIT rev-parse --git-dir >/dev/null 2>&1 \
    || die "$REPO 不是 git 工作树 —— 本脚本必须在题目容器里跑"
  for f in "${SCOPE[@]}"; do
    [ -e "$REPO/$f" ] || die "被计分单元不存在:$REPO/$f(REPO 不对?可用 KFC_REPO_DIR 覆盖)"
  done
  HEAD_BEFORE="$(GIT rev-parse HEAD)" || die "读不到 $REPO 的 HEAD"
  N_BEFORE="$(GIT rev-list --count HEAD)"
  DIRTY_BEFORE="$(GIT status --porcelain -- "${SCOPE[@]}" | sed '/^$/d')"
  GIT checkout -q HEAD -- "${SCOPE[@]}" || die "复位被计分单元失败(${SCOPE[*]})"
  DIRTY_AFTER="$(GIT status --porcelain -- "${SCOPE[@]}" | sed '/^$/d')"
  if [ -n "$DIRTY_AFTER" ]; then
    printf '%s\n' "$DIRTY_AFTER" | sed "s|^|[solve.sh $TASK_ID]   |" >&2
    die "复位后被计分单元仍不干净(见上)—— 量不到真正的 no-op 值"
  fi
  [ "$HEAD_BEFORE" = "$(GIT rev-parse HEAD)" ] || die "HEAD 被移动了,判分会把 candidate 当基线"
  [ "$N_BEFORE" = "$(GIT rev-list --count HEAD)" ] || die "commit 数变了,HEAD 不再是唯一的挖洞基线"
  if [ -n "$DIRTY_BEFORE" ]; then
    say "--noop:已把可编辑面复位到烤入基线;原先这些是脏的:"
    printf '%s\n' "$DIRTY_BEFORE" | sed "s|^|[solve.sh $TASK_ID]   |"
  else
    say "--noop:可编辑面本来就是干净的烤入基线,无需复位"
  fi
  say "--noop 完成:${SCOPE[*]} = 烤入基线(HEAD 未动 @ ${HEAD_BEFORE:0:12})"
  exit 0
fi

cat >&2 <<'EOF'
本题**不随包提供参考实现** —— solution/ 下只有 README.md,没有 oracle / negative /
baseline2 产物(见 solution/README.md:参考实现只存在于单文件变体形式,未随本包发布)。

因此本脚本无法落地任何参考实现:
  · 被计分单元 : /app/repo/offload_policy.py
    (来源 tests/test.sh 的 SCOPE=("offload_policy.py"),$REPO=/app/repo)
  · verifier 本身仍原生支持 oracle / negative / noop 模式分派 —— 参考实现按路径传入、
    从不烤进镜像。你补上一份等价实现后,重标定通路可直接跑通:
      docker exec -e KERNELBENCH_VERIFY_MODE=oracle \
                  -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
                  <容器> bash /tests/test.sh
  · tests/ref_speedup.txt 里的锚点是建题环境标定的(见 tests/ref_speedup.caveat.md),
    换硬件后不可比,且没有可直接执行的重标定通路。

exit 2 = 本题不支持 solve.sh 落地参考实现(不是脚本出错);--noop 仍然 exit 0。
EOF
exit 2
