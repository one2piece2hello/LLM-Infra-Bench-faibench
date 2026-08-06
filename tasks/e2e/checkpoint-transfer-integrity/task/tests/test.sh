#!/usr/bin/env bash
# Verifier for correctness-e2e-e5-checkpoint-transfer-integrity (IMPL-CLASS, perf_metric:none).
# Medium-topic E5 (CROSS.STORAGE.TRANSFER / CHECKPOINT / INTEGRITY). The candidate implements a
# checkpoint-transfer manager: pack->chunks+parity+manifest, all-or-nothing durable upload, verifying
# + resumable download, XOR-parity reconstruction of one missing chunk, artifact-path normalization.
# reward = BINARY (reward.md 实现类): 1.0 iff EVERY graded case passes and no cheat/gate
# trips, else 0.0; passed/total + per-axis detail are emitted for offline diagnosis only. Hard-fail gates
# (frozen-surface touch, missing entry, banned-path hardcode, harness crash) force 0.0.
#
# MODE dispatch (supplement C): candidate | oracle | negative.
set -uo pipefail
git config --global --add safe.directory '*' 2>/dev/null || true
export PATH=/opt/kernelbench-venv/bin:$PATH
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB=/app/submission
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
export E2E_SUBMISSION_DIR="$SUB"

HARD=(); add_hard(){ HARD+=("$1"); }
scope_ok=true; entry_ok=true; sourcescan_ok=true

# MODE 注入（reviewer 侧，评分时新鲜上传，绝不进镜像）：优先整文件替换 *_FILE，回退 *_PATCH。
if [ "$MODE" = "oracle" ]; then
  if [ -n "${KERNELBENCH_ORACLE_FILE:-}" ]; then
    cp "$KERNELBENCH_ORACLE_FILE" "$SUB/ckpt_transfer.py" || add_hard "oracle_copy_failed"
  elif [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ]; then
    git -C "$SUB" apply -p1 "$KERNELBENCH_ORACLE_PATCH" 2>"$LOG/oracle_apply.log" || add_hard "oracle_apply_failed"
  fi
fi
if [ "$MODE" = "negative" ]; then
  if [ -n "${KERNELBENCH_NEGATIVE_FILE:-}" ]; then
    cp "$KERNELBENCH_NEGATIVE_FILE" "$SUB/ckpt_transfer.py" || true
  elif [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
    git -C "$SUB" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true
  fi
fi

# ---- frozen-surface hard gate: candidate edits ONLY /app/submission/ckpt_transfer.py ----
if [ -d "$SUB/.git" ]; then
  git -C "$SUB" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    case "$f" in
      ckpt_transfer.py) : ;;
      *__pycache__*|*.pyc) : ;;
      *) scope_ok=false; add_hard "out_of_scope_edit:$f" ;;
    esac
  done < "$LOG/changed_files.txt"
fi
for bad in workload.py test.sh reward.json; do
  [ -e "$SUB/$bad" ] && { scope_ok=false; add_hard "forbidden_file_in_submission:$bad"; }
done

[ -f "$SUB/ckpt_transfer.py" ] || { entry_ok=false; add_hard "entry_missing:ckpt_transfer.py"; }

# ---- source scan: forbid verifier-path hardcoding + importing the harness / reference ----
if [ -f "$SUB/ckpt_transfer.py" ]; then
  python3 - "$SUB/ckpt_transfer.py" > "$LOG/source_scan.log" 2>&1 <<'PY'
import sys
src = open(sys.argv[1]).read()
bad = []
for tok in ("/tests", "/opt/verifier", "/logs/verifier", "reward.json", "correctness_trace",
            "E2E_RESULT", "_ref_chunks", "_ref_parity", "_ref_normalize", "_Store", "_DownSink",
            "import workload", "from workload"):
    if tok in src:
        bad.append(tok)
if bad:
    print("BANNED", bad); sys.exit(3)
print("CLEAN"); sys.exit(0)
PY
  [ $? -eq 3 ] && { sourcescan_ok=false; add_hard "banned_path_or_import"; }
fi

# ---- CORRECTNESS: run graded cases with an overall watchdog (concurrency cases could hang) ----
reward=0.0
tests_passed=0
tests_total=0
frac_diag=0.0
test_fail_reason=""
if [ "$scope_ok" = true ] && [ "$entry_ok" = true ] && [ "$sourcescan_ok" = true ]; then
  ( cd "$TESTS_DIR" && timeout 1200 python3 workload.py ) > "$LOG/correctness.out" 2>&1 || true
  rm -f "$LOG/_reward" "$LOG/_frac" "$LOG/_passed" "$LOG/_total" "$LOG/_testfail"
  python3 - > "$LOG/correctness_frac.log" 2>&1 <<'PY'
import json, sys
L = "/logs/verifier"
trace = None
try:
    for l in open(L + "/correctness.out"):
        if l.startswith("E2E_RESULT "):
            trace = json.loads(l[len("E2E_RESULT "):]).get("correctness_trace"); break
except Exception as e:
    print("READ_ERR", e)
if trace is None:
    print("NO_TRACE")
    # 5-file 输出契约（输出契约）：harness 崩溃/无 trace 时也必须留下 correctness_results.json
    json.dump({"completed": False, "passed": 0, "total": 0, "correctness_frac": 0.0,
               "by_axis": {}, "cases": [], "error": "harness_no_trace"},
              open(L + "/correctness_results.json", "w"))
    open(L + "/_reward", "w").write("0.0")
    open(L + "/_frac", "w").write("0.0")
    open(L + "/_passed", "w").write("0")
    open(L + "/_total", "w").write("0")
    open(L + "/_testfail", "w").write("harness_no_trace")
    sys.exit(3)
completed = bool(trace.get("completed"))
passed = int(trace.get("passed") or 0)
total = int(trace.get("total") or 0)
# diagnostic ONLY -- reward.md 实现类 reward 是二值，pass-rate 不再缩放分数
frac = float(trace.get("correctness_frac", 0.0)) if completed else 0.0
binary = 1.0 if (completed and total > 0 and passed == total) else 0.0
json.dump(trace, open(L + "/correctness_results.json", "w"))
open(L + "/_reward", "w").write(repr(binary))
open(L + "/_frac", "w").write(repr(frac))
open(L + "/_passed", "w").write(str(passed))
open(L + "/_total", "w").write(str(total))
if binary != 1.0:
    reason = ("harness_incomplete" if not completed
              else "no_graded_cases" if total == 0
              else "tests_failed:%d/%d" % (passed, total))
    open(L + "/_testfail", "w").write(reason)
print("BINARY", binary, "frac", frac, "passed", passed, "total", total,
      "by_axis", trace.get("by_axis"))
sys.exit(0 if completed else 3)
PY
  [ -f "$LOG/_reward" ] && reward="$(cat "$LOG/_reward")"
  [ -f "$LOG/_passed" ] && tests_passed="$(cat "$LOG/_passed")"
  [ -f "$LOG/_total" ]  && tests_total="$(cat "$LOG/_total")"
  [ -f "$LOG/_frac" ]   && frac_diag="$(cat "$LOG/_frac")"
  [ -f "$LOG/_testfail" ] && test_fail_reason="$(cat "$LOG/_testfail")"
else
  test_fail_reason="gate_hard_fail_before_case_run"
fi

nhard="${#HARD[@]}"; [ "$nhard" -gt 0 ] && reward=0.0
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
# reward.md 归零条件汇总：门失败 ∪ 有测例未通过
reasons_str="$hard_str"
if [ -n "$test_fail_reason" ]; then reasons_str="$reasons_str $test_fail_reason"; fi

export E2E_MODE="$MODE" E2E_REWARD="$reward" E2E_HARD="$hard_str" E2E_REASONS="$reasons_str" \
       E2E_PASSED="$tests_passed" E2E_TOTAL="$tests_total" E2E_FRAC="$frac_diag" \
       E2E_SCOPE="$scope_ok" E2E_ENTRY="$entry_ok" E2E_SRC="$sourcescan_ok"
python3 - <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return 0.0
def i(x):
    try: return int(x)
    except: return 0
reward_val = f(os.environ.get("E2E_REWARD"))
# 实现类 reward 只能是 0.0 / 1.0（reward.md）；任何异常值一律归 0
if reward_val not in (0.0, 1.0):
    reward_val = 0.0
hard = [h for h in os.environ.get("E2E_HARD", "").split() if h]
reasons = [h for h in os.environ.get("E2E_REASONS", "").split() if h]
if reasons and reward_val != 0.0:
    reward_val = 0.0
# 归零必须有命名 reason（契约要求）：reward==0 而 reasons 为空是一个可观测缺陷
if reward_val == 0.0 and not reasons:
    reasons = ["zero_without_named_reason"]
tests = {"passed": i(os.environ.get("E2E_PASSED")), "total": i(os.environ.get("E2E_TOTAL"))}
verifier_state = {"mode": os.environ.get("E2E_MODE"), "task_kind": "correctness",
                  "task_type": "implementation", "perf_metric": "none",
                  "reward_form": "binary: 1.0 iff all graded cases pass and no gate/cheat trip, else 0.0",
                  "gates": {"scope_ok": os.environ.get("E2E_SCOPE") == "true",
                            "entry_ok": os.environ.get("E2E_ENTRY") == "true",
                            "sourcescan_ok": os.environ.get("E2E_SRC") == "true"},
                  "tests": tests, "hard_fails": hard, "hard_fail_reasons": reasons}
json.dump(verifier_state, open("/logs/verifier/verifier_state.json", "w"))
reward = {"task_type": "implementation", "reward": reward_val, "hard_fail_reasons": reasons,
          "tests": tests,
          # ---- 以下仅供离线诊断，不参与 reward 计算 ----
          "correctness_frac": f(os.environ.get("E2E_FRAC")), "task_kind": "correctness",
          "perf_metric": "none", "mode": os.environ.get("E2E_MODE"), "hard_fails": hard}
json.dump(reward, open("/logs/verifier/reward.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(reward_val))
# 第 5 个契约文件：实现类无计时，但契约要求 5 个文件都存在（输出契约）
json.dump({"task_type": "implementation", "reward": reward_val, "task_kind": "correctness",
           "perf_metric": "none", "mode": os.environ.get("E2E_MODE"), "tests": tests,
           "hard_fail_reasons": reasons, "timing_measured": False,
           "note": "implementation-class task: no timing is measured; emitted to satisfy the 5-file verifier output contract"},
          open("/logs/verifier/benchmark_results.json", "w"))
print(json.dumps(reward))
PY

# ---- 5-file 输出契约的 shell 兜底（输出契约）------------------------------------------
# 上面的 python 段若整段崩溃（解释器缺失 / OOM / 磁盘满），历史上会产出 0 个契约文件，harness 侧
# 只能看到"无结果"。这里用纯 shell 补齐缺失的文件，归零并给出命名 reason。已存在且非空的文件不动。
_FB_REASON="verifier_contract_fallback"
_missing=""
for _f in reward.txt reward.json verifier_state.json correctness_results.json benchmark_results.json; do
  [ -s "$LOG/$_f" ] || _missing="$_missing $_f"
done
if [ -n "$_missing" ]; then
  echo "VERIFIER_CONTRACT_FALLBACK missing:$_missing" >&2
  case "$_missing" in *reward.json*) printf '0.0' > "$LOG/reward.txt" ;; esac
  [ -s "$LOG/reward.txt" ] || printf '0.0' > "$LOG/reward.txt"
  [ -s "$LOG/reward.json" ] || printf '%s' \
    "{\"task_type\":\"implementation\",\"reward\":0.0,\"hard_fail_reasons\":[\"$_FB_REASON\"],\"tests\":{\"passed\":0,\"total\":0},\"task_kind\":\"correctness\",\"perf_metric\":\"none\",\"mode\":\"$MODE\"}" \
    > "$LOG/reward.json"
  [ -s "$LOG/verifier_state.json" ] || printf '%s' \
    "{\"mode\":\"$MODE\",\"task_kind\":\"correctness\",\"task_type\":\"implementation\",\"perf_metric\":\"none\",\"reward_form\":\"binary: 1.0 iff all graded cases pass and no gate/cheat trip, else 0.0\",\"tests\":{\"passed\":0,\"total\":0},\"hard_fails\":[\"$_FB_REASON\"],\"hard_fail_reasons\":[\"$_FB_REASON\"]}" \
    > "$LOG/verifier_state.json"
  [ -s "$LOG/correctness_results.json" ] || printf '%s' \
    "{\"completed\":false,\"passed\":0,\"total\":0,\"correctness_frac\":0.0,\"by_axis\":{},\"cases\":[],\"error\":\"$_FB_REASON\"}" \
    > "$LOG/correctness_results.json"
  [ -s "$LOG/benchmark_results.json" ] || printf '%s' \
    "{\"task_type\":\"implementation\",\"reward\":0.0,\"task_kind\":\"correctness\",\"perf_metric\":\"none\",\"mode\":\"$MODE\",\"tests\":{\"passed\":0,\"total\":0},\"hard_fail_reasons\":[\"$_FB_REASON\"],\"timing_measured\":false}" \
    > "$LOG/benchmark_results.json"
fi
echo "VERIFIER_CONTRACT_FILES=$(ls -1 "$LOG"/reward.txt "$LOG"/reward.json "$LOG"/verifier_state.json "$LOG"/correctness_results.json "$LOG"/benchmark_results.json 2>/dev/null | wc -l)/5"
exit 0
