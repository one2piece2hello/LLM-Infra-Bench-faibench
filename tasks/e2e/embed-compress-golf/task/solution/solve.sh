#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/embed-compress-golf
#   embedding compression golf: best nDCG@10 under a 64 B/vector budget (CPU-scored)
#
# ORACLE FORM      : SINGLE FILE (whole-file replacement of the submission encoder).
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/submission)
# GRADED UNIT      : /app/submission/submission_encoder.py (+ optional encoder_config.json)
# PRIMARY METRIC   : nDCG@10 quality ratio -> min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, ref_speedup=1.4290238
# MODE VARIABLE    : VERIFIER_MODE (candidate|strong_baseline|negative). The non-candidate modes copy
#                    $STRONG_BASELINE_DIR/$NEGATIVE_DIR/submission_encoder.py into the submission dir;
#                    solve.sh does the same copy, so grade in the default candidate mode.
# REFERENCE ASSETS : solution/submission_encoder_ceiling_ref.py (ceiling / ref_speedup),
#                    solution/submission_encoder_ref.py (strong baseline = the 1.0 anchor)
# SHA-PINNED PATHS : tests/verifier-correctness-manifest.json: compute_reward_sha256 +
#                    heldout_assets.sha256{heldout_corpus.jsonl,heldout_queries.jsonl,heldout_qrels.json},
#                    and environment/tests_heldout_sha256.txt. All of them live under tests/ or
#                    environment/ -- solve.sh writes ONLY under $SUBMISSION_DIR.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/e2e-g2-embed-compress-golf:oss bash -lc '
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
#                    HOW, here: the image bakes ONLY submission_encoder_template.py -- there is no
#                    submission_encoder.py until something writes one, so the literal pristine state is an
#                    entry-contract hard fail, not a measurement. --noop therefore installs that TEMPLATE
#                    as the submission: that IS the model-visible no-op floor. (If even the template is
#                    absent it removes the entry file and says so.)
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
#
# NOTE: the image bakes only /app/submission/submission_encoder_template.py -- there is no
#       submission_encoder.py until something writes one, so the literal pristine state is an
#       entry-contract hard fail. --noop therefore installs the model-visible TEMPLATE as the
#       submission (that is the real no-op floor) and says so.
# =============================================================================
set -uo pipefail

TASK_ID='embed-compress-golf'
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
  printf '  %-26s %s\n' '(default) / --oracle' 'submission_encoder_ceiling_ref.py -- PCA-32 + fp16 + fp refiner (~0.5)'
  printf '  %-26s %s\n' '--baseline2' 'submission_encoder_ref.py -- the strong-baseline 1.0 anchor (scores ~0)'
  printf '  %-26s %s\n' '--noop' 'install the baked starter TEMPLATE as the submission -- the no-op floor'
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
  oracle)    land 'solution/submission_encoder_ceiling_ref.py=>submission_encoder.py' ;;
  baseline2) land 'solution/submission_encoder_ref.py=>submission_encoder.py' ;;
  noop)      if ! unsnap "$SUB/submission_encoder.py"; then
               if [ -f "$SUB/submission_encoder_template.py" ]; then
                 cp -f "$SUB/submission_encoder_template.py" "$SUB/submission_encoder.py" \
                   && say "no-op arm = the baked starter TEMPLATE installed as submission_encoder.py"
               else
                 rm -f "$SUB/submission_encoder.py"
                 say "removed submission_encoder.py: the pristine image ships only the template,"
                 say "so the literal no-op arm is an entry-contract hard fail (reward 0)."
               fi
             fi ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # candidate mode (default); scoring is pinned to CPU by test.sh'
