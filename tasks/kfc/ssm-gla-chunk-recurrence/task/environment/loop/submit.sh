#!/usr/bin/env bash
# Per-task 3-to-16-submission loop harness for
# ssm-gla-chunk-recurrence.
#
# The solver's ONLY sanctioned interface to scoring during the session
# (0755 root:root; the engine + private assets it calls are 0700 root-owned).
# Baked at /opt/loop/submit.sh so it is present from the first turn — unlike
# /tests/, which the runner creates only AFTER the agent finishes.
#
# Usage:
#   bash /opt/loop/submit.sh                      # score current /app/repo, get feedback
#   bash /opt/loop/submit.sh --finalize [--reason "..."]   # plant best iterate, end loop
#
# Each `submit`:
#   1. runs the baked DEV engine (visible-subset correctness + dev timing) on
#      the LIVE /app/repo and returns SANITIZED feedback;
#   2. snapshots the candidate tree (git stash-create tree hash + binary diff)
#      to /logs/loop/submissions/<k>/;
#   3. updates /logs/loop/best.json (best correctness-passing dev_reward);
#   4. enforces min=3 / max=16.
# On --finalize (>=3) or auto at k=16 it restores /app/repo's WORKING TREE to
# the best correctness-passing snapshot (HEAD stays at the baseline commit so
# the runner's end-of-session verifier still sees a product-only modification)
# and writes /logs/loop/finalize.json for the audit.
#
# 🔴 PATH-pin: the agent shell + runner verifier exec are
# NON-login; bare python3 has no torch. Pin the venv FIRST.
export PATH=/opt/kernelbench-venv/bin:/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
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
FINALIZE_JSON="$LOOP/finalize.json"

_read_count() {
  if [ -f "$COUNT_FILE" ]; then
    awk 'NR==1{gsub(/[^0-9]/,""); if($0=="")$0="0"; print; exit}' "$COUNT_FILE"
  else
    echo 0
  fi
}

# Non-mutating snapshot of the current working tree -> a git tree hash.
# `git stash create` builds dangling objects without touching HEAD/index/WT.
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
  # Restore /app/repo working tree to the best correctness-passing snapshot.
  git -C "$REPO" reset -q --hard HEAD 2>/dev/null || true
  git -C "$REPO" clean -qfd 2>/dev/null || true
  WIN_DIFF=""
  WIN_SUB=""
  if [ -f "$BEST_FILE" ]; then
    WIN_SUB="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("submission",""))' "$BEST_FILE" 2>/dev/null || true)"
  fi
  if [ -n "$WIN_SUB" ] && [ -f "$LOOP/submissions/$WIN_SUB/candidate.diff" ]; then
    WIN_DIFF="$LOOP/submissions/$WIN_SUB/candidate.diff"
    if [ -s "$WIN_DIFF" ]; then
      git -C "$REPO" apply --whitespace=nowarn "$WIN_DIFF" 2>"$LOOP/finalize_apply.log" || \
        echo "WARNING: could not re-apply the winning diff cleanly; see /logs/loop/finalize_apply.log" >&2
    fi
  fi
  # Write finalize.json (winning index, k, dev_speedup trajectory, reason) for audit.
  REASON="$REASON" WIN_SUB="$WIN_SUB" CUR="$CUR" python3 - <<'PY'
import json, os
from pathlib import Path
LOOP = Path('/logs/loop')
best = {}
try:
    best = json.loads((LOOP / 'best.json').read_text())
except Exception:
    best = {}
traj = []
try:
    for line in (LOOP / 'state.jsonl').read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        traj.append({'submission': row.get('submission'),
                     'dev_speedup': row.get('speedup'),
                     'dev_reward': row.get('reward'),
                     'correctness_ok': row.get('correctness_ok')})
except Exception:
    pass
win = best.get('submission') if best else None
out = {
    'schema_version': 'kernelbench_loop_finalize_v1',
    'task_id': 'ssm-gla-chunk-recurrence',
    'mode': 'baked',
    'submissions_used': int(os.environ.get('CUR') or 0),
    'min_submissions': 1, 'max_submissions': 1,
    'winning_submission': win,
    'winning_dev_speedup': best.get('speedup') if best else None,
    'winning_dev_reward': best.get('reward') if best else None,
    'winning_tree': best.get('tree') if best else None,
    'dev_speedup_trajectory': traj,
    'termination_reason': os.environ.get('REASON') or '',
    'note': 'dev_reward is an in-session proxy; meta.reward is the runner end-of-session score of /app/repo.',
}
(LOOP / 'finalize.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
PY
  # Integrity: recompute the restored tree hash for the audit trail.
  RESTORED_TREE="$(_snapshot_tree)"
  echo "restored_tree=$RESTORED_TREE" >> "$LOOP/finalize_apply.log" 2>/dev/null || true
  if [ -n "$WIN_SUB" ]; then
    WR="$(python3 -c 'import json,sys;b=json.load(open(sys.argv[1]));print(f"{b.get(\"speedup\")}", f"{b.get(\"reward\")}")' "$BEST_FILE" 2>/dev/null || echo "? ?")"
    echo "finalized: submission $WIN_SUB at k=$CUR/$MAX_SUBMISSIONS (dev_speedup ${WR% *}, dev_reward ${WR#* }); the grader will now score /app/repo"
  else
    echo "finalized: no correctness-passing submission in $CUR tries — /app/repo restored to the starting baseline; final reward will be 0 (no_improvement). the grader will now score /app/repo"
  fi
  exit 0
fi

# ------------------- submit -------------------
CUR="$(_read_count)"; CUR="${CUR:-0}"
if [ "$CUR" -ge "$MAX_SUBMISSIONS" ]; then
  echo "submission $CUR/$MAX_SUBMISSIONS"
  echo "budget exhausted — call 'bash /opt/loop/submit.sh --finalize' to grade your best submission"
  # Auto-finalize as a safety net.
  echo "--- auto-finalizing best-of-$MAX_SUBMISSIONS ---"
  exec bash "$HERE/submit.sh" --finalize --reason "auto-finalize at budget ceiling"
fi

# Run the DEV engine on the LIVE /app/repo.
bash "$HERE/score_engine.sh"
engine_rc=$?

# HARNESS error path: the engine flagged an infra failure (rc=3 or a marker).
# Do NOT consume the solver's budget for our bug.
if [ "$engine_rc" -eq 3 ] || [ -f "$DEV/harness_error.txt" ]; then
  python3 "$HERE/sanitize_feedback.py"
  exit 0
fi

# Consume one submission.
NEXT=$((CUR + 1))
echo "$NEXT" > "$COUNT_FILE"

# Snapshot the candidate: tree hash (non-mutating) + binary diff vs baseline HEAD.
SNAP="$LOOP/submissions/$NEXT"
mkdir -p "$SNAP"
TREE="$(_snapshot_tree)"
git -C "$REPO" diff --binary HEAD > "$SNAP/candidate.diff" 2>/dev/null || true
echo "${TREE:-}" > "$SNAP/tree.txt"
for f in verifier_state.json correctness_results.json benchmark_results.json reward.json; do
  [ -f "$DEV/$f" ] && cp "$DEV/$f" "$SNAP/$f" 2>/dev/null || true
done

# Update best-of-k (only correctness-passing, strictly-greater dev_reward) and
# append the per-submission audit row.
NEXT="$NEXT" TREE="$TREE" python3 - <<'PY'
import json, os, time
from pathlib import Path
LOOP = Path('/logs/loop'); DEV = LOOP / 'dev'
k = int(os.environ['NEXT']); tree = os.environ.get('TREE') or ''
try:
    state = json.loads((DEV / 'verifier_state.json').read_text())
except Exception:
    state = {}
try:
    reward = json.loads((DEV / 'reward.json').read_text())
except Exception:
    reward = {}
corr_ok = bool(state.get('correctness_ok'))
r = float(reward.get('reward') or 0.0)
s = float(reward.get('speedup') or 0.0)
# append audit row
row = {'submission': k, 'ts': int(time.time()), 'correctness_ok': corr_ok,
       'reward': r, 'speedup': s, 'tree': tree,
       'hard_fail_reasons': list(state.get('hard_fail_reasons') or [])}
with (LOOP / 'state.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(row) + '\n')
# update best
if corr_ok:
    best_p = LOOP / 'best.json'
    try:
        best = json.loads(best_p.read_text())
    except Exception:
        best = None
    if best is None or r > float(best.get('reward') or -1.0):
        best_p.write_text(json.dumps(
            {'submission': k, 'reward': r, 'speedup': s, 'tree': tree},
            indent=2, sort_keys=True) + '\n')
PY

# Print sanitized feedback.
python3 "$HERE/sanitize_feedback.py"

# Auto-finalize on the ceiling.
if [ "$NEXT" -ge "$MAX_SUBMISSIONS" ]; then
  echo ""
  echo "--- budget exhausted (submission $MAX_SUBMISSIONS/$MAX_SUBMISSIONS). Auto-finalizing best-of-$MAX_SUBMISSIONS. ---"
  exec bash "$HERE/submit.sh" --finalize --reason "auto-finalize at budget ceiling"
fi

exit 0
