#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/a4-token-efficiency-budget
#   LM training under a harness-owned TOKEN budget (nanoGPT, H20)
#
# ORACLE FORM      : DIRECTORY holding ONE file. solution/<variant>/train_gpt.py is copied to
#                    $SUBMISSION_DIR/train_gpt.py.
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/submission)
# GRADED UNIT      : /app/submission/train_gpt.py -- must expose train_for_budget(seed, out_dir,
#                    budget_seconds) and load_model_for_verification(ckpt, device)
# PRIMARY METRIC   : median val_bpb ratio at a fixed token budget -> min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
# MODE VARIABLE    : VERIFIER_MODE exists but is a TRAP: tests/test.sh step 1 hard-fails with
#                    anti_cheat_triggered for any VERIFIER_MODE that is set and != candidate
#                    (mode dispatch was deliberately removed -- the solver owns the environment).
#                    So: land the recipe and grade with VERIFIER_MODE UNSET.
# REFERENCE ASSETS : solution/{ceiling,strong_baseline,naive,negative,probe_budget_bite}/train_gpt.py
# SHA-PINNED PATHS : tests/verifier-correctness-manifest.json:frozen_surface = /tests/{compute_reward.py,
#                    test.sh,holdout_corpus.txt,verifier-correctness-manifest.json} + compute_reward_sha256
#                    + data_loader_checksum.sha256. solve.sh writes ONLY under $SUBMISSION_DIR.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/e2e-a4-token-efficiency-budget:oss-loop16 bash -lc '
#       bash /task/solution/solve.sh && bash /tests/test.sh'
#   -> /logs/verifier/reward.json
#
# CLI: (default)=oracle | --negative | --baseline2 | --noop | --variant <name> | --list | --help
#
# --noop SEMANTICS : --noop 把可编辑面复位到烤入基线（与 verifier 取基线的方式一致，HEAD 不动），
#                    使对照组能量到真正的 no-op 值。
#                    (`--noop` resets the editable surface to the baked baseline -- the same way the verifier
#                    materialises its own timing/scoring baseline -- with HEAD untouched, so the control arm
#                    measures the REAL no-op value instead of whatever a previous arm left behind.)
#                    HOW, here: rm -f $SUB/model_ckpt.pt, then restore train_gpt.py from
#                    the pre-landing snapshot under ${SOLVE_BACKUP_DIR:-/tmp/.fai_solve_baseline}/<task-id>/
#                    (here /app/submission is NOT git-tracked -- only /app/repo is).
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
# =============================================================================
set -uo pipefail

TASK_ID='a4-token-efficiency-budget'
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
  printf '  %-26s %s\n' '(default) / --oracle' 'ceiling -- the in-budget ceiling (the ref_speedup anchor, ~0.5)'
  printf '  %-26s %s\n' '--baseline2' 'strong_baseline -- the tuned-AdamW 1.0 anchor (scores ~0)'
  printf '  %-26s %s\n' '--negative' 'negative -- the budget-cheating recipe, must score 0'
  printf '  %-26s %s\n' '--variant naive' 'the model-visible naive starter'
  printf '  %-26s %s\n' '--variant probe_budget_bite' 'probe: does the token budget really bind?'
  printf '  %-26s %s\n' '--noop' 'reset to the baked starter train_gpt.py -- the TRUE no-op arm'
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
  oracle)    land 'solution/ceiling/train_gpt.py=>train_gpt.py' ;;
  baseline2) land 'solution/strong_baseline/train_gpt.py=>train_gpt.py' ;;
  negative)  land 'solution/negative/train_gpt.py=>train_gpt.py' ;;
  naive|probe_budget_bite)
             land "solution/$MODE/train_gpt.py=>train_gpt.py" ;;
  noop)      rm -f "$SUB/model_ckpt.pt"; restore train_gpt.py ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # 🔴 do NOT export VERIFIER_MODE -- any non-candidate value hard-fails'
