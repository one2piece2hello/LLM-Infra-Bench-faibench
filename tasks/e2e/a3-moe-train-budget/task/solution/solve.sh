#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/a3-moe-train-budget
#   MoE pretraining under a harness-owned wall-clock budget (nanoGPT, H20)
#
# ORACLE FORM      : DIRECTORY. solution/<variant>/ holds a complete 2-file recipe
#                    (run_training.sh + train_gpt.py) that is copied into the submission dir.
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/submission)
# GRADED UNIT      : /app/submission/run_training.sh + /app/submission/train_gpt.py
#                    (the verifier runs run_training.sh under its own timer and grades the
#                    checkpoint it writes to $SUBMISSION_DIR/model_ckpt.pt)
# PRIMARY METRIC   : val_bpb ratio at a fixed wall-clock budget -> min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
# MODE VARIABLE    : NONE. tests/test.sh has NO mode dispatch: it strips every inherited E2E_*
#                    variable and unsets VERIFIER_MODE / KERNELBENCH_VERIFY_MODE, then grades
#                    /app/submission and nothing else. So the ONLY way to score a reference is to
#                    land it into the submission dir -- which is exactly what this script does.
# REFERENCE ASSETS : solution/{oracle_grouped_moe,mid_naive_moe,baseline_dense_floor,
#                    probe_budget_bite,probe_no_checkpoint,probe_tamper_frozen}/
# SHA-PINNED PATHS : tests/verifier-correctness-manifest.json:frozen_surface = /tests/{compute_reward.py,
#                    test.sh,holdout_corpus.txt,timer.sh,verifier-correctness-manifest.json} +
#                    compute_reward_sha256. test.sh sha256s EVERY file in /tests before the training
#                    and re-checks after. solve.sh writes ONLY under $SUBMISSION_DIR -- it never
#                    touches /tests, so no pin is disturbed.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/e2e-a3-moe-train-budget:oss bash -lc '
#       bash /task/solution/solve.sh && bash /tests/test.sh'
#   -> /logs/verifier/reward.json
#
# CLI: (default)=oracle | --baseline2 | --noop | --variant <name> | --list | --help
#
# --noop SEMANTICS : --noop 把可编辑面复位到烤入基线（与 verifier 取基线的方式一致，HEAD 不动），
#                    使对照组能量到真正的 no-op 值。
#                    (`--noop` resets the editable surface to the baked baseline -- the same way the verifier
#                    materialises its own timing/scoring baseline -- with HEAD untouched, so the control arm
#                    measures the REAL no-op value instead of whatever a previous arm left behind.)
#                    HOW, here: rm -f $SUB/model_ckpt.pt, then restore run_training.sh + train_gpt.py from
#                    the pre-landing snapshot under ${SOLVE_BACKUP_DIR:-/tmp/.fai_solve_baseline}/<task-id>/
#                    (here /app/submission is NOT git-tracked -- only /app/repo is).
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
#
# NOTE: the harness runs run_training.sh under an unprivileged uid with /tests chmod-000, so the
#       recipe must not depend on anything under /tests. RECIPE_RECORD.json in the oracle dir is
#       documentation only and is deliberately NOT landed (it would add an untracked file to the
#       submission surface for no scoring benefit).
# =============================================================================
set -uo pipefail

TASK_ID='a3-moe-train-budget'
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$SELF")"
BK="${SOLVE_BACKUP_DIR:-/tmp/.fai_solve_baseline}/${TASK_ID}"

say()  { printf '[solve:%s] %s\n' "$TASK_ID" "$*"; }
warn() { printf '[solve:%s] WARN: %s\n' "$TASK_ID" "$*" >&2; }
die()  { printf '[solve:%s] ERROR: %s\n' "$TASK_ID" "$*" >&2; exit 1; }

# Reference assets live OUTSIDE the image on purpose. Look for them wherever the
# operator may have mounted the package; SOLVE_ASSET_DIR overrides everything.
ASSET_ROOTS=("${SOLVE_ASSET_DIR:-}" "$SELF" "$PKG" "$PKG/tests" "$PKG/solution" \
             /solution /solution/tests /task /task/solution /task/tests /tests /opt/solution)
asset() { local rel="$1" r
  for r in "${ASSET_ROOTS[@]}"; do [ -n "$r" ] || continue
    [ -e "$r/$rel" ] && { printf '%s\n' "$r/$rel"; return 0; }
  done
  return 1
}
need() { asset "$1" || die "reference asset '$1' not found.
  searched: ${ASSET_ROOTS[*]}
  mount the task package (e.g. -v <pkg>:/task:ro) or set SOLVE_ASSET_DIR=<dir>."; }

# Pristine snapshot, kept OUTSIDE every graded surface (/tmp), so --noop can undo a
# landing inside a long-lived container. Taken once, right before the first overwrite.
_k() { printf '%s' "$1" | tr '/' '_'; }
snap() { local f="$1" k; k="$(_k "$1")"; mkdir -p "$BK" 2>/dev/null || return 0
  [ -e "$BK/$k" ] || [ -e "$BK/$k.ABSENT" ] || {
    if [ -e "$f" ]; then cp -p "$f" "$BK/$k" 2>/dev/null || true
    else : > "$BK/$k.ABSENT" 2>/dev/null || true; fi; }; }
unsnap() { local f="$1" k; k="$(_k "$1")"
  if   [ -e "$BK/$k" ];        then cp -p "$BK/$k" "$f" && say "restored $f from the pristine snapshot"
  elif [ -e "$BK/$k.ABSENT" ]; then rm -f "$f" && say "removed $f (absent in the pristine image)"
  else return 1; fi; }

put() { # put <src> <dst> : idempotent copy, snapshots the pristine <dst> first
  local s="$1" d="$2"
  [ -f "$s" ] || die "missing reference file: $s"
  mkdir -p "$(dirname "$d")" || die "cannot create $(dirname "$d")"
  snap "$d"
  if cmp -s "$s" "$d"; then say "already landed (identical): $d"
  else cp -f "$s" "$d" || die "copy failed: $s -> $d"; say "landed $(basename "$s") -> $d"; fi
  [ -x "$s" ] && { chmod +x "$d" 2>/dev/null || true; }
  return 0
}

head_of() { git -C "$1" rev-parse HEAD 2>/dev/null || echo NO_GIT; }
assert_head() { # assert_head <dir> <before>
  local now; now="$(head_of "$1")"
  [ "$now" = "$2" ] || die "HEAD of $1 moved ($2 -> $now).
  The verifier grades the WORK TREE and asserts HEAD is still the single baked baseline
  commit (RUNNABLE_SPEC H3) -- a commit makes a correct solution score 0."
  [ "$now" = NO_GIT ] || say "HEAD unchanged: $1 @ $(printf '%.12s' "$now")"
}

MODE=oracle
while [ $# -gt 0 ]; do
  case "$1" in
    --oracle)    MODE=oracle ;;
    --negative)  MODE=negative ;;
    --baseline2) MODE=baseline2 ;;
    --noop)      MODE=noop ;;
    --variant)   shift; [ $# -gt 0 ] || die "--variant needs a name"; MODE="$1" ;;
    --list)      MODE=__list ;;
    -h|--help)   MODE=__help ;;
    *)           die "unknown argument '$1' (try --help)" ;;
  esac
  shift
done

list_variants() {
  printf 'variants for %s:\n' "$TASK_ID"
  printf '  %-26s %s\n' '(default) / --oracle' 'oracle_grouped_moe -- the in-budget ceiling (ref_speedup anchor)'
  printf '  %-26s %s\n' '--baseline2' 'baseline_dense_floor -- the dense 1.0 anchor (scores ~0)'
  printf '  %-26s %s\n' '--variant mid_naive_moe' 'the naive MoE mid-tier arm'
  printf '  %-26s %s\n' '--variant probe_budget_bite' 'probe: does the wall-clock kill really bite?'
  printf '  %-26s %s\n' '--variant probe_no_checkpoint' 'probe: no checkpoint -> entry-contract hard fail'
  printf '  %-26s %s\n' '--variant probe_tamper_frozen' 'probe: touches /tests -> anti_cheat_triggered'
  printf '  %-26s %s\n' '--noop' 'reset to the baked starter recipe -- the TRUE no-op arm'
}
if [ "$MODE" = __list ]; then list_variants; exit 0; fi
if [ "$MODE" = __help ]; then
  sed -n '2,/^set -uo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d'
  list_variants
  exit 0
fi

SUB="${SUBMISSION_DIR:-/app/submission}"
GITDIR='NONE'

land() { # land <asset-relpath=>dest-name> ...
  local before spec s d
  [ -d "$SUB" ] || mkdir -p "$SUB" || die "cannot create $SUB"
  before="$(head_of "$GITDIR")"
  for spec in "$@"; do
    s="${spec%%=>*}"; d="${spec##*=>}"
    put "$(need "$s")" "$SUB/$d"
  done
  assert_head "$GITDIR" "$before"
  [ "$before" = NO_GIT ] || { say "work-tree state of $GITDIR:"
    git -C "$GITDIR" status --porcelain 2>/dev/null | sed 's/^/    /'; }
}

restore() { # restore <dest-name> ...
  local before f; before="$(head_of "$GITDIR")"
  for f in "$@"; do
    if [ "$before" != NO_GIT ] \
       && git -C "$GITDIR" ls-files --error-unmatch -- "$SUB/$f" >/dev/null 2>&1; then
      git -C "$GITDIR" checkout -- "$SUB/$f" \
        && say "restored $SUB/$f from the baked baseline commit"
    elif ! unsnap "$SUB/$f"; then
      warn "no pristine copy of $SUB/$f is recoverable here -- use a fresh container for a true no-op arm"
    fi
  done
  assert_head "$GITDIR" "$before"
}

case "$MODE" in
  oracle)    land 'solution/oracle_grouped_moe/run_training.sh=>run_training.sh' \
                  'solution/oracle_grouped_moe/train_gpt.py=>train_gpt.py' ;;
  baseline2) land 'solution/baseline_dense_floor/run_training.sh=>run_training.sh' \
                  'solution/baseline_dense_floor/train_gpt.py=>train_gpt.py' ;;
  mid_naive_moe|probe_budget_bite|probe_no_checkpoint|probe_tamper_frozen)
             land "solution/$MODE/run_training.sh=>run_training.sh" \
                  "solution/$MODE/train_gpt.py=>train_gpt.py" ;;
  noop)      rm -f "$SUB/model_ckpt.pt"; restore run_training.sh train_gpt.py ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # no mode variable; test.sh strips E2E_*/VERIFIER_MODE itself'
