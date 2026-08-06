#!/usr/bin/env bash
# ============================================================================
# wro loop16 finalizer — submit.sh  (REUSABLE, task-agnostic)
# ============================================================================
# The solver's ONLY sanctioned scoring interface during a session
# (0755 root:root; the engine + private assets it calls are 0700 root-owned).
# Baked at /opt/loop/submit.sh so it exists from the first turn — unlike
# /tests/, which the grading harness creates only AFTER the agent finishes.
#
#   bash /opt/loop/submit.sh                    # score current /app/repo, get feedback
#   bash /opt/loop/submit.sh --finalize [--reason "..."]   # plant best iterate, end loop
#
# A wro submission is the solver's EDITS to the declared /app/repo scope files,
# scored by the task's own tests/test.sh in candidate mode -> perf_metric.
# Budget: exactly ONE graded submission (kfc is single-submission). The first
# submit.sh scores and immediately auto-finalizes; there is no second scored attempt.
# end-of-session /tests/test.sh scores the best-of-k iterate. HEAD stays at the
# baked baseline commit, so the anti-cheat scope-diff sees a product-only change.
#
# 🔴 git-safe-directory: a root-owned /app/repo is
# root-owned; without this every git op inside test.sh fails "dubious ownership"
# and silently returns a false 1.0. MUST be the first effective line.
git config --global --add safe.directory '*' 2>/dev/null || true

# 🔴 PATH-pin: the agent shell + runner verifier exec are NON-login; bare python3
# may lack the task's numpy/torch. Pin the venv (if present) first, then system.
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOOP=/logs/loop
DEV="$LOOP/dev"
mkdir -p "$LOOP" "$LOOP/submissions" "$DEV"

MIN_SUBMISSIONS=1
MAX_SUBMISSIONS=1
COUNT_FILE="$LOOP/count"
BEST_FILE="$LOOP/best.json"
STATE_JSONL="$LOOP/state.jsonl"

# task_id for the finalize record — read from the baked private manifest (generic).
TASK_ID="$(python3 -c 'import json;print(json.load(open("/opt/loop/private/manifest.json")).get("task_id",""))' 2>/dev/null || true)"
[ -n "$TASK_ID" ] || TASK_ID="wro-loop16"

_read_count() {
  if [ -f "$COUNT_FILE" ]; then
    awk 'NR==1{gsub(/[^0-9]/,""); if($0=="")$0="0"; print; exit}' "$COUNT_FILE"
  else
    echo 0
  fi
}

# Non-mutating snapshot of the current working tree -> a git tree hash.
_snapshot_tree() {
  local c
  c="$(git -C "$REPO" stash create 2>/dev/null || true)"
  if [ -n "$c" ]; then
    git -C "$REPO" rev-parse "$c^{tree}" 2>/dev/null || git -C "$REPO" rev-parse "HEAD^{tree}" 2>/dev/null
  else
    git -C "$REPO" rev-parse "HEAD^{tree}" 2>/dev/null
  fi
}

# ------------------- argument parsing -------------------
MODE=submit
REASON=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --finalize) MODE=finalize; shift ;;
    --reason) REASON="${2:-}"; shift 2 ;;
    --reason=*) REASON="${1#--reason=}"; shift ;;
    *) echo "usage: bash /opt/loop/submit.sh [--finalize [--reason \"<text>\"]]" >&2; exit 2 ;;
  esac
done

# ------------------- finalize -------------------
if [ "$MODE" = finalize ]; then
  CUR="$(_read_count)"; CUR="${CUR:-0}"
  if [ "$CUR" -lt "$MIN_SUBMISSIONS" ]; then
    echo "minimum $MIN_SUBMISSIONS submission(s) required before finalize (you have $CUR/$MIN_SUBMISSIONS)"
    exit 1
  fi
  # Reset the working tree to the baked baseline (HEAD), then plant the winner.
  git -C "$REPO" reset -q --hard HEAD 2>/dev/null || true
  git -C "$REPO" clean -qfd 2>/dev/null || true
  WIN_SUB=""; WIN_TREE=""
  if [ -f "$BEST_FILE" ]; then
    WIN_SUB="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("submission",""))' "$BEST_FILE" 2>/dev/null || true)"
    WIN_TREE="$(python3 -c 'import json,sys;b=json.load(open(sys.argv[1]));print(b.get("tree_hash") or b.get("tree") or "")' "$BEST_FILE" 2>/dev/null || true)"
  fi
  PLANTED="none"
  if [ -n "$WIN_TREE" ] && git -C "$REPO" read-tree "$WIN_TREE" 2>"$LOOP/finalize_apply.log"; then
    # read-tree loaded the winning tree into the index; write it to the worktree.
    # HEAD is unchanged (still the baseline commit) -> anti-cheat sees product-only diff.
    git -C "$REPO" checkout-index -f -a 2>>"$LOOP/finalize_apply.log" && PLANTED="read-tree"
  fi
  if [ "$PLANTED" = "none" ] && [ -n "$WIN_SUB" ] && [ -s "$LOOP/submissions/$WIN_SUB/candidate.diff" ]; then
    # fallback: re-apply the winning binary diff onto the baseline.
    git -C "$REPO" apply --whitespace=nowarn "$LOOP/submissions/$WIN_SUB/candidate.diff" 2>>"$LOOP/finalize_apply.log" \
      && PLANTED="diff-apply" \
      || echo "WARNING: could not plant the winning submission; see /logs/loop/finalize_apply.log" >&2
  fi
  REASON="$REASON" WIN_SUB="$WIN_SUB" WIN_TREE="$WIN_TREE" CUR="$CUR" PLANTED="$PLANTED" TASK_ID="$TASK_ID" python3 - <<'PY'
import json, os
from pathlib import Path
LOOP = Path('/logs/loop')
try: best = json.loads((LOOP / 'best.json').read_text())
except Exception: best = {}
traj = []
try:
    for line in (LOOP / 'state.jsonl').read_text().splitlines():
        line = line.strip()
        if not line: continue
        row = json.loads(line)
        traj.append({'submission': row.get('submission'), 'dev_speedup': row.get('speedup'),
                     'dev_reward': row.get('reward'), 'correctness_ok': row.get('correctness_ok')})
except Exception: pass
out = {
    'schema_version': 'kernelbench_loop_finalize_v1',
    'task_id': os.environ.get('TASK_ID') or 'wro-loop16',
    'mode': 'baked', 'plant_method': os.environ.get('PLANTED'),
    'submissions_used': int(os.environ.get('CUR') or 0),
    'min_submissions': 1, 'max_submissions': 1,
    'winning_submission': (best.get('submission') if best else None),
    'winning_tree': os.environ.get('WIN_TREE') or None,
    'winning_dev_speedup': (best.get('speedup') if best else None),
    'winning_dev_reward': (best.get('reward') if best else None),
    'dev_speedup_trajectory': traj,
    'termination_reason': os.environ.get('REASON') or '',
    'note': 'dev_reward is an in-session proxy; meta.reward is the runner end-of-session score of /app/repo.',
}
(LOOP / 'finalize.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
PY
  RESTORED_TREE="$(_snapshot_tree)"
  echo "restored_tree=$RESTORED_TREE plant=$PLANTED" >> "$LOOP/finalize_apply.log" 2>/dev/null || true
  if [ -n "$WIN_SUB" ] && [ "$PLANTED" != "none" ]; then
    echo "finalized: submission $WIN_SUB at k=$CUR/$MAX_SUBMISSIONS (planted via $PLANTED); the grader will now score /app/repo"
  else
    echo "finalized: no correctness-passing submission in $CUR tries — /app/repo restored to the starting baseline; final reward will be the unchanged-baseline score (~1.0). the grader will now score /app/repo"
  fi
  exit 0
fi

# ------------------- submit -------------------
CUR="$(_read_count)"; CUR="${CUR:-0}"
if [ "$CUR" -ge "$MAX_SUBMISSIONS" ]; then
  echo "submission $CUR/$MAX_SUBMISSIONS"
  echo "budget exhausted — call 'bash /opt/loop/submit.sh --finalize' to grade your best submission"
  echo "--- auto-finalizing best-of-$MAX_SUBMISSIONS ---"
  exec bash "$HERE/submit.sh" --finalize --reason "auto-finalize at budget ceiling"
fi

# Run the DEV engine on the LIVE /app/repo.
bash "$HERE/score_engine.sh"
engine_rc=$?

# HARNESS error path: infra failure (rc=3 or a marker). Do NOT consume budget.
if [ "$engine_rc" -eq 3 ] || [ -f "$DEV/harness_error.txt" ]; then
  python3 "$HERE/sanitize_feedback.py"
  exit 0
fi

NEXT=$((CUR + 1))
echo "$NEXT" > "$COUNT_FILE"

SNAP="$LOOP/submissions/$NEXT"
mkdir -p "$SNAP"
TREE="$(_snapshot_tree)"
git -C "$REPO" diff --binary HEAD > "$SNAP/candidate.diff" 2>/dev/null || true
echo "${TREE:-}" > "$SNAP/tree_hash.txt"
echo "${TREE:-}" > "$SNAP/tree.txt"
for f in verifier_state.json reward.json benchmark_results.json; do
  [ -f "$DEV/$f" ] && cp "$DEV/$f" "$SNAP/$f" 2>/dev/null || true
done

NEXT="$NEXT" TREE="$TREE" python3 - <<'PY'
import json, os, time
from pathlib import Path
LOOP = Path('/logs/loop'); DEV = LOOP / 'dev'
k = int(os.environ['NEXT']); tree = os.environ.get('TREE') or ''
try: state = json.loads((DEV / 'verifier_state.json').read_text())
except Exception: state = {}
try: reward = json.loads((DEV / 'reward.json').read_text())
except Exception: reward = {}
corr_ok = bool(state.get('correctness_ok'))
r = float(reward.get('reward') or 0.0); s = float(reward.get('speedup') or 0.0)
row = {'submission': k, 'ts': int(time.time()), 'correctness_ok': corr_ok,
       'reward': r, 'speedup': s, 'tree': tree,
       'hard_fail_reasons': list(state.get('hard_fail_reasons') or [])}
with (LOOP / 'state.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(row) + '\n')
if corr_ok:
    best_p = LOOP / 'best.json'
    try: best = json.loads(best_p.read_text())
    except Exception: best = None
    if best is None or r > float(best.get('reward') or -1.0):
        best_p.write_text(json.dumps({'submission': k, 'reward': r, 'speedup': s, 'tree': tree, 'tree_hash': tree},
                                     indent=2, sort_keys=True) + '\n')
PY

python3 "$HERE/sanitize_feedback.py"

if [ "$NEXT" -ge "$MAX_SUBMISSIONS" ]; then
  echo ""
  echo "--- budget exhausted (submission $MAX_SUBMISSIONS/$MAX_SUBMISSIONS). Auto-finalizing best-of-$MAX_SUBMISSIONS. ---"
  exec bash "$HERE/submit.sh" --finalize --reason "auto-finalize at budget ceiling"
fi

exit 0
