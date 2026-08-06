#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/varlen-prefill-attn-sol
#   varlen causal prefill attention kernel (H20)
#
# ORACLE FORM      : SINGLE FILE (whole-file replacement of the submission entry module).
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/repo/submission) inside the 1-commit git tree /app/repo
# GRADED UNIT      : $SUBMISSION_DIR/varlen_prefill_attn.py (/app/submission is a symlink to it)
# PRIMARY METRIC   : geomean SOL fraction vs the strong baseline -> min(1, ln(speedup/1.5317)/ln(1.5317)), 0 unless speedup > 1.5317
# MODE VARIABLE    : VERIFIER_MODE (candidate|strong_baseline|negative_*|ceiling_*). The non-candidate
#                    modes resolve $NEGATIVE_DIR/<mode>.py with several name fallbacks, so they work if
#                    you mount solution/ as $NEGATIVE_DIR. Landing into the submission dir is simpler:
#                    grade in candidate mode.
# REFERENCE ASSETS : solution/ceiling_triton_prefill.py (= manifest oracle_impl), solution/negative_{bigmem,
#                    mutate,nowrite,stale,window}.py, solution/control_{flex_attention,sdpa_perseq}.py
# SHA-PINNED PATHS : tests/verifier-correctness-manifest.json:frozen_surface_sha256 covers
#                    compute_reward.py, test.sh, harness/bench_prefill.py, harness/baseline_prefill.py
#                    (and hidden_suite.json) -- all under tests/. solve.sh writes ONLY under
#                    $SUBMISSION_DIR.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/e2e-d1-varlen:oss bash -lc '
#       bash /task/solution/solve.sh && bash /tests/test.sh'
#   -> /logs/verifier/reward.json
#
# CLI: (default)=oracle | --negative | --noop | --variant <name> | --list | --help
#
# --noop SEMANTICS : --noop 把可编辑面复位到烤入基线（与 verifier 取基线的方式一致，HEAD 不动），
#                    使对照组能量到真正的 no-op 值。
#                    (`--noop` resets the editable surface to the baked baseline -- the same way the verifier
#                    materialises its own timing/scoring baseline -- with HEAD untouched, so the control arm
#                    measures the REAL no-op value instead of whatever a previous arm left behind.)
#                    HOW, here: git -C ${REPO_DIR:-/app/repo} checkout -- $SUB/varlen_prefill_attn.py.
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
# =============================================================================
set -uo pipefail

TASK_ID='varlen-prefill-attn-sol'
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
  printf '  %-26s %s\n' '(default) / --oracle' 'ceiling_triton_prefill.py -- the reviewer ceiling (ref_speedup anchor)'
  printf '  %-26s %s\n' '--negative' 'negative_mutate.py -- mutates the callers q in place, must score 0'
  printf '  %-26s %s\n' '--variant negative_bigmem' 'over-allocating negative (expect 0)'
  printf '  %-26s %s\n' '--variant negative_nowrite' 'does not write the output (expect 0)'
  printf '  %-26s %s\n' '--variant negative_stale' 'stale-cache negative (expect 0)'
  printf '  %-26s %s\n' '--variant negative_window' 'wrong causal window (expect 0)'
  printf '  %-26s %s\n' '--variant control_flex_attention' 'off-the-shelf flex_attention control (must not beat the baseline)'
  printf '  %-26s %s\n' '--variant control_sdpa_perseq' 'off-the-shelf per-sequence SDPA control'
  printf '  %-26s %s\n' '--noop' 'reset the entry to the baked baseline -- the TRUE no-op arm'
}
if [ "$MODE" = __list ]; then list_variants; exit 0; fi
if [ "$MODE" = __help ]; then
  sed -n '2,/^set -uo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d'
  list_variants
  exit 0
fi

SUB="${SUBMISSION_DIR:-/app/repo/submission}"
GITDIR="${REPO_DIR:-/app/repo}"

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
  oracle)    land 'solution/ceiling_triton_prefill.py=>varlen_prefill_attn.py' ;;
  negative)  land 'solution/negative_mutate.py=>varlen_prefill_attn.py' ;;
  negative_bigmem|negative_mutate|negative_nowrite|negative_stale|negative_window|\
control_flex_attention|control_sdpa_perseq)
             land "solution/$MODE.py=>varlen_prefill_attn.py" ;;
  noop)      restore varlen_prefill_attn.py ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # candidate mode (default)'
