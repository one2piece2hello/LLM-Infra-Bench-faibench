#!/usr/bin/env bash
# ============================================================================
# e2e-a3-moe-train-budget — loop16 submit.sh  (protocol 1..16, model decides)
# ============================================================================
# The solver's ONLY sanctioned in-session scoring interface. 0755 root:root; the engine
# and private dev assets it calls are 0700 root-owned (unreadable by the solver's uid).
#
#   bash /opt/loop/submit.sh                    # score current state, get feedback
#   bash /opt/loop/submit.sh --finalize [--reason "..."]   # plant best-of-k, end the loop
#
# Budget: min=1, max=16. --finalize is accepted at ANY k>=1 (there is no minimum and no
# "k consecutive rounds without gain" rule); at k=16 the harness auto-finalizes.
#
# Each round is scored by /opt/loop/score_engine.sh on a PUBLIC dev proxy (a small public
# token budget + a first-principles correctness self-check): the overlay contains NO held-out
# corpus, NO reference recipe and NO calibrated anchor, so the per-round dev_val_bpb is a
# PUBLIC PROXY, not the graded score. The GRADED score is produced after the session by the
# runner from a freshly-uploaded /tests over the HIDDEN test split, against whatever this
# script plants.
#
# a4 has TWO editable graded surfaces — /app/submission (the entry contract the grader loads)
# and /app/repo (the nanoGPT the recipe imports LIVE at grading). Best-of-k therefore snapshots
# and plants BOTH: /app/repo via git tree-hash (read-tree + checkout-index), /app/submission via
# a per-round tarball. On finalize the BEST correctness-passing snapshot (not the last edit) is
# planted; /app/repo HEAD stays at the baked baseline commit.

# git-safe-directory: a crane-appended /app/repo is root-owned; without this every git op
# fails "dubious ownership". MUST be first.
git config --global --add safe.directory '*' 2>/dev/null || true

# PATH-pin: agent shell + runner exec are NON-login; pin the venv so python has torch.
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
SUBM=/app/submission
BASE_SUBM=/opt/loop/baseline_submission     # pristine shipped starter (baked)
LOOP=/logs/loop
DEV="$LOOP/dev"
SUBS="$LOOP/submissions"
mkdir -p "$LOOP" "$SUBS" "$DEV"

MIN_SUBMISSIONS=1
MAX_SUBMISSIONS=16
COUNT_FILE="$LOOP/count"
BEST_FILE="$LOOP/best.json"
STATE_JSONL="$LOOP/state.jsonl"

TASK_ID="$(python3 -c 'import json;print(json.load(open("/opt/loop/private/manifest.json")).get("task_id",""))' 2>/dev/null || true)"
[ -n "$TASK_ID" ] || TASK_ID="e2e-a3-moe-train-budget"

_read_count() {
  if [ -f "$COUNT_FILE" ]; then
    awk 'NR==1{gsub(/[^0-9]/,""); if($0=="")$0="0"; print; exit}' "$COUNT_FILE"
  else
    echo 0
  fi
}

# git tree-hash of the current /app/repo working tree (non-mutating index add).
_repo_tree() {
  git -C "$REPO" add -A >/dev/null 2>&1 || true
  git -C "$REPO" write-tree 2>/dev/null || git -C "$REPO" rev-parse "HEAD^{tree}" 2>/dev/null || echo ""
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
    echo "nothing to finalize yet: score at least $MIN_SUBMISSIONS submission first (you have $CUR)."
    echo "Run: bash /opt/loop/submit.sh"
    exit 1
  fi

  WIN_SUB=""; WIN_REPO_TREE=""
  if [ -f "$BEST_FILE" ]; then
    WIN_SUB="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("submission",""))' "$BEST_FILE" 2>/dev/null || true)"
    WIN_REPO_TREE="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("repo_tree","") or "")' "$BEST_FILE" 2>/dev/null || true)"
  fi

  # --- plant /app/repo: reset to baked baseline HEAD, then load the winning tree ---
  git -C "$REPO" reset -q --hard HEAD 2>/dev/null || true
  git -C "$REPO" clean -qfd 2>/dev/null || true
  PLANT_REPO="none"
  if [ -n "$WIN_REPO_TREE" ] && git -C "$REPO" read-tree "$WIN_REPO_TREE" 2>"$LOOP/finalize_apply.log"; then
    git -C "$REPO" checkout-index -f -a 2>>"$LOOP/finalize_apply.log" && PLANT_REPO="read-tree"
  fi
  if [ "$PLANT_REPO" = "none" ] && [ -n "$WIN_SUB" ] && [ -s "$SUBS/$WIN_SUB/repo.diff" ]; then
    git -C "$REPO" apply --whitespace=nowarn "$SUBS/$WIN_SUB/repo.diff" 2>>"$LOOP/finalize_apply.log" \
      && PLANT_REPO="diff-apply" || true
  fi

  # --- plant /app/submission: extract the winning tarball (or restore pristine starter) ---
  PLANT_SUBM="none"
  if [ -n "$WIN_SUB" ] && [ -s "$SUBS/$WIN_SUB/submission.tgz" ]; then
    rm -rf "${SUBM:?}/"* 2>/dev/null || true
    mkdir -p "$SUBM"
    if tar xzf "$SUBS/$WIN_SUB/submission.tgz" -C "$SUBM" 2>>"$LOOP/finalize_apply.log"; then
      PLANT_SUBM="tar"
    fi
  fi
  if [ "$PLANT_SUBM" = "none" ]; then
    # no correctness-passing snapshot -> restore the pristine shipped starter
    rm -rf "${SUBM:?}/"* 2>/dev/null || true
    mkdir -p "$SUBM"
    if [ -d "$BASE_SUBM" ]; then
      cp -a "$BASE_SUBM/." "$SUBM/" 2>>"$LOOP/finalize_apply.log" && PLANT_SUBM="baseline_restore"
    fi
  fi

  REASON="$REASON" WIN_SUB="$WIN_SUB" WIN_REPO_TREE="$WIN_REPO_TREE" CUR="$CUR" \
  PLANT_REPO="$PLANT_REPO" PLANT_SUBM="$PLANT_SUBM" TASK_ID="$TASK_ID" python3 - <<'PY'
import json, os, hashlib
from pathlib import Path
LOOP = Path('/logs/loop')
try: best = json.loads((LOOP / 'best.json').read_text())
except Exception: best = {}
traj = []
try:
    for line in (LOOP / 'state.jsonl').read_text().splitlines():
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        traj.append({'submission': r.get('submission'),
                     'dev_val_bpb': r.get('dev_val_bpb'),
                     'correctness_ok': r.get('correctness_ok')})
except Exception: pass
# sha of the planted entry, so best-of-k can be PROVEN (not just claimed)
planted_sha = None
try:
    planted_sha = hashlib.sha256(Path('/app/submission/train_gpt.py').read_bytes()).hexdigest()
except Exception:
    planted_sha = None
out = {
    'schema_version': 'kb_loop_finalize_v1',
    'task_id': os.environ.get('TASK_ID') or 'e2e-a3-moe-train-budget',
    'mode': 'baked',
    'plant_method': {'submission': os.environ.get('PLANT_SUBM'), 'repo': os.environ.get('PLANT_REPO')},
    'submissions_used': int(os.environ.get('CUR') or 0),
    'min_submissions': 1, 'max_submissions': 16,
    'winning_submission': (best.get('submission') if best else None),
    'winning_repo_tree': os.environ.get('WIN_REPO_TREE') or None,
    'winning_dev_val_bpb': (best.get('dev_val_bpb') if best else None),
    'planted_train_gpt_sha256': planted_sha,
    'dev_public_proxy_trajectory': traj,
    'termination_reason': os.environ.get('REASON') or '',
    'note': 'dev_val_bpb is an in-session PUBLIC-proxy on a small public token budget, used ONLY to rank best-of-k; the authoritative score is the runner end-of-session grade of /app/submission + /app/repo over the hidden test split.',
}
(LOOP / 'finalize.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
PY

  if [ -n "$WIN_SUB" ] && { [ "$PLANT_SUBM" = "tar" ] || [ "$PLANT_REPO" != "none" ]; }; then
    echo "finalized: submission $WIN_SUB at k=$CUR/$MAX_SUBMISSIONS (submission planted via $PLANT_SUBM, repo via $PLANT_REPO); the grader will now score /app/submission + /app/repo"
  else
    echo "finalized: no correctness-passing submission in $CUR tries — the shipped starter was restored; the grader will now score it (under this task's bounded reward, a model that does not beat the dense-at-floor starter scores 0)"
  fi
  exit 0
fi

# ------------------- submit -------------------
CUR="$(_read_count)"; CUR="${CUR:-0}"
if [ "$CUR" -ge "$MAX_SUBMISSIONS" ]; then
  echo "submission $CUR/$MAX_SUBMISSIONS"
  echo "budget exhausted — auto-finalizing best-of-$MAX_SUBMISSIONS"
  exec bash "$HERE/submit.sh" --finalize --reason "auto-finalize at budget ceiling"
fi

# Run the DEV engine on the LIVE surfaces.
bash "$HERE/score_engine.sh"
engine_rc=$?

# HARNESS-ERROR refund: infra failure (rc=3 or marker) -> do NOT consume budget.
if [ "$engine_rc" -eq 3 ] || [ -f "$DEV/harness_error.txt" ]; then
  python3 "$HERE/sanitize_feedback.py"
  exit 0
fi

NEXT=$((CUR + 1))
echo "$NEXT" > "$COUNT_FILE"

SNAP="$SUBS/$NEXT"
mkdir -p "$SNAP"
REPO_TREE="$(_repo_tree)"
echo "${REPO_TREE:-}" > "$SNAP/repo_tree.txt"
git -C "$REPO" diff --binary HEAD > "$SNAP/repo.diff" 2>/dev/null || true
tar czf "$SNAP/submission.tgz" -C "$SUBM" . 2>/dev/null || true
for f in verifier_state.json reward.json; do
  [ -f "$DEV/$f" ] && cp "$DEV/$f" "$SNAP/$f" 2>/dev/null || true
done

NEXT="$NEXT" REPO_TREE="$REPO_TREE" python3 - <<'PY'
import json, os, time
from pathlib import Path
LOOP = Path('/logs/loop'); DEV = LOOP / 'dev'
k = int(os.environ['NEXT']); repo_tree = os.environ.get('REPO_TREE') or ''
try: state = json.loads((DEV / 'verifier_state.json').read_text())
except Exception: state = {}
try: reward = json.loads((DEV / 'reward.json').read_text())
except Exception: reward = {}
corr_ok = bool(state.get('correctness_ok'))
dev_reward = float(reward.get('dev_score') or 0.0)      # higher is better (= 1/val_bpb)
dev_val_bpb = reward.get('dev_val_bpb')
row = {'submission': k, 'ts': int(time.time()), 'correctness_ok': corr_ok,
       'dev_score': dev_reward, 'dev_val_bpb': dev_val_bpb, 'repo_tree': repo_tree,
       'failing_invariant': state.get('failing_invariant') or ''}
with (LOOP / 'state.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(row) + '\n')
if corr_ok:
    bp = LOOP / 'best.json'
    try: best = json.loads(bp.read_text())
    except Exception: best = None
    if best is None or dev_reward > float(best.get('dev_score') or -1.0):
        bp.write_text(json.dumps({'submission': k, 'dev_score': dev_reward,
                                  'dev_val_bpb': dev_val_bpb, 'repo_tree': repo_tree},
                                 indent=2, sort_keys=True) + '\n')
PY

python3 "$HERE/sanitize_feedback.py"

if [ "$NEXT" -ge "$MAX_SUBMISSIONS" ]; then
  echo ""
  echo "--- budget exhausted (submission $MAX_SUBMISSIONS/$MAX_SUBMISSIONS). Auto-finalizing best-of-$MAX_SUBMISSIONS. ---"
  exec bash "$HERE/submit.sh" --finalize --reason "auto-finalize at budget ceiling"
fi
exit 0
