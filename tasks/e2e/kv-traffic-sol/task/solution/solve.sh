#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/kv-traffic-sol
#   paged-KV traffic engine: fused per-op kernels at bit-exact round-trip parity (H20)
#
# ORACLE FORM      : SINGLE FILE (whole-file replacement of the submission entry module).
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/repo/submission) inside the 1-commit git tree /app/repo
# GRADED UNIT      : $SUBMISSION_DIR/kv_traffic.py (/app/submission is a symlink to /app/repo/submission)
# PRIMARY METRIC   : geomean SOL fraction vs the strong baseline -> min(1, ln(speedup/2.5799321)/ln(2.5799321)), 0 unless speedup > 2.5799321
# MODE VARIABLE    : VERIFIER_MODE (candidate|strong_baseline|negative_*|ceiling_*). The non-candidate
#                    modes are resolved INSIDE tests/compute_reward.py as
#                    $NEGATIVE_DIR/<mode-minus-first-token>.py -- which does NOT match the shipped
#                    file names (negative_alias_no_store -> looks for alias_no_store.py) and silently
#                    falls back to grading the CANDIDATE. Landing into the submission dir, as solve.sh
#                    does, side-steps that entirely: grade in candidate mode.
# REFERENCE ASSETS : solution/ceiling_triton_fused.py (= manifest oracle_impl), solution/negative_{alias_no_store,
#                    double_layout,lossy_fp8,partial_write,stale_plan}.py, solution/_kb_base.py
# SHA-PINNED PATHS : tests/verifier-correctness-manifest.json:frozen_surface_sha256 covers
#                    compute_reward.py, test.sh, harness/bench_kvtraffic.py, harness/baseline_kv_traffic.py,
#                    harness/hidden_suite.json -- all under tests/. solve.sh writes ONLY under
#                    $SUBMISSION_DIR and never touches tests/.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/b1-kv-traffic-sol:oss bash -lc '
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
#                    HOW, here: rm -f $SUB/_kb_base.py (a negative sidecar), then
#                    git -C ${REPO_DIR:-/app/repo} checkout -- $SUB/kv_traffic.py.
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
#
# NOTE: every negative_*.py does `from _kb_base import BaseEngine`, and the harness loads the
#       entry with importlib WITHOUT adding its directory to sys.path (it also cd's to /tmp).
#       So --negative also lands _kb_base.py next to the entry and prints the PYTHONPATH the
#       negative arm needs. The oracle has no such dependency.
# =============================================================================
set -uo pipefail

TASK_ID='kv-traffic-sol'
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
  printf '  %-26s %s\n' '(default) / --oracle' 'ceiling_triton_fused.py -- the reviewer ceiling (ref_speedup anchor)'
  printf '  %-26s %s\n' '--negative' 'negative_partial_write.py + _kb_base.py (must score 0)'
  printf '  %-26s %s\n' '--variant negative_alias_no_store' 'aliasing negative (+ _kb_base.py)'
  printf '  %-26s %s\n' '--variant negative_double_layout' 'layout negative (+ _kb_base.py)'
  printf '  %-26s %s\n' '--variant negative_lossy_fp8' 'lossy-fp8 negative (+ _kb_base.py)'
  printf '  %-26s %s\n' '--variant negative_stale_plan' 'stale-plan negative (+ _kb_base.py)'
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
  oracle)    land 'solution/ceiling_triton_fused.py=>kv_traffic.py' ;;
  negative)  MODE=negative_partial_write
             land "solution/$MODE.py=>kv_traffic.py" 'solution/_kb_base.py=>_kb_base.py' ;;
  negative_alias_no_store|negative_double_layout|negative_lossy_fp8|negative_partial_write|negative_stale_plan)
             land "solution/$MODE.py=>kv_traffic.py" 'solution/_kb_base.py=>_kb_base.py' ;;
  noop)      rm -f "$SUB/_kb_base.py"; restore kv_traffic.py ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # candidate mode (default)'
say '  the negatives need their shared base importable:'
say '    PYTHONPATH="$SUBMISSION_DIR" bash /tests/test.sh'
