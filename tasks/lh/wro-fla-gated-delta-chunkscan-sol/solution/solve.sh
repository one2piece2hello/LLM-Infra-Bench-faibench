#!/usr/bin/env bash
# =============================================================================
# solve.sh -- lh/wro-fla-gated-delta-chunkscan-sol
#   fla gated-delta-rule chunk-scan (3-file scope), H20
#
# ORACLE FORM      : PATCH (git apply -p1 onto the editable repo work tree)
# EDITABLE SURFACE : $REPO_DIR (default /app/repo), a 1-commit git tree
# GRADED UNIT      : fla/ops/gated_delta_rule/chunk.py, fla/ops/gated_delta_rule/chunk_fwd.py, fla/ops/gated_delta_rule/gate.py
# PRIMARY METRIC   : speedup (performance)
# MODE VARIABLE    : KERNELBENCH_VERIFY_MODE (candidate|noop|oracle|negative)
#                    solve.sh lands the reference INTO THE WORK TREE, so you grade in the
#                    DEFAULT candidate mode -- do NOT also set KERNELBENCH_VERIFY_MODE=oracle
#                    (that would make test.sh apply the patch a second time and fail).
# REFERENCE ASSETS : oracle.patch, negative.patch
# LOOP16           : YES (environment/loop -> /opt/loop). solve.sh covers the SINGLE-SHOT oracle arm;
#                    the loop16 arm is driven by bash /opt/loop/submit.sh [--finalize].
# SHA-PINNED PATHS : none in this package (no frozen_surface_sha256 / .frozen_hashes.json).
#                    tests/** and environment/** are read-only for solve.sh regardless.
#
# GRADE AFTER LANDING (one command, from the package root):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/lh-fla-gated-delta-chunkscan:oss bash -lc '
#       bash /task/solution/solve.sh && bash /tests/test.sh'
#   -> /logs/verifier/reward.json  (reward.txt, verifier_state.json, ... alongside)
#
# CLI: (default)=oracle | --negative | --noop | --variant <name> | --list | --help
#
# --noop SEMANTICS : --noop 把可编辑面复位到烤入基线（与 verifier 取基线的方式一致，HEAD 不动），
#                    使对照组能量到真正的 no-op 值。
#                    (`--noop` resets the editable surface to the baked baseline -- the same way the verifier
#                    materialises its own timing/scoring baseline -- with HEAD untouched, so the control arm
#                    measures the REAL no-op value instead of whatever a previous arm left behind.)
#                    HOW, here: git -C $REPO_DIR checkout -- \
#                        fla/ops/gated_delta_rule/chunk.py \
#                        fla/ops/gated_delta_rule/chunk_fwd.py \
#                        fla/ops/gated_delta_rule/gate.py
#
# HEAD MUST NOT MOVE. This script only edits the WORK TREE -- no git add, no commit, no
# branch. The verifier reads `git status --porcelain` for its scope gate and materialises
# the timing baseline from HEAD, so a commit turns a correct solution into reward 0
# (RUNNABLE_SPEC H3). The script re-reads HEAD after landing and aborts if it moved.
# =============================================================================
set -uo pipefail

TASK_ID='wro-fla-gated-delta-chunkscan-sol'
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
  printf '  %-26s %s\n' '(default) / --oracle' 'oracle.patch -- the reference solution'
  printf '  %-26s %s\n' '--negative' 'negative.patch -- known-bad, must score 0'
  printf '  %-26s %s\n' '--noop' 'reset the scope files to the baked baseline -- the TRUE no-op arm'
}
if [ "$MODE" = __list ]; then list_variants; exit 0; fi
if [ "$MODE" = __help ]; then
  sed -n '2,/^set -uo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d'
  list_variants
  exit 0
fi

REPO="${REPO_DIR:-/app/repo}"
SCOPE=("fla/ops/gated_delta_rule/chunk.py" "fla/ops/gated_delta_rule/chunk_fwd.py" "fla/ops/gated_delta_rule/gate.py")
EXTRA_CLEAN=()

land_patch() { # land_patch <patch> <label>
  local p="$1" lbl="$2" before err
  [ -d "$REPO" ] || die "editable repo not found at $REPO (set REPO_DIR)"
  git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || warn "$REPO is not a git work tree -- the scope gate and the baseline timing will misbehave"
  before="$(head_of "$REPO")"
  if git -C "$REPO" apply --check -R -p1 "$p" >/dev/null 2>&1; then
    say "$lbl is ALREADY applied in $REPO (reverse check passes) -- idempotent no-op"
  else
    err="$(git -C "$REPO" apply --check -p1 "$p" 2>&1)" || {
      printf '%s\n' "$err" >&2
      die "$lbl does not apply onto $REPO.
  The tree is not the pristine baked baseline. Run '--noop' first, or use a fresh container."
    }
    git -C "$REPO" apply -p1 --whitespace=nowarn "$p" || die "$lbl failed to apply"
    say "applied $lbl (git apply -p1) -> $REPO"
  fi
  assert_head "$REPO" "$before"
  say "work-tree state after landing:"
  git -C "$REPO" status --porcelain 2>/dev/null | sed 's/^/    /'
}

restore_baseline() {
  local before; before="$(head_of "$REPO")"
  [ -d "$REPO" ] || die "editable repo not found at $REPO"
  git -C "$REPO" checkout -- "${SCOPE[@]}" 2>/dev/null \
    || warn "git checkout of the scope files failed (not a git tree?)"
  local f
  for f in ${EXTRA_CLEAN[@]+"${EXTRA_CLEAN[@]}"}; do
    [ -n "$f" ] && [ -e "$REPO/$f" ] && rm -f "$REPO/$f" && say "removed $REPO/$f"
  done
  assert_head "$REPO" "$before"
  say "no-op arm: the scope files are back at the baked baseline. Residual work-tree diff:"
  git -C "$REPO" status --porcelain 2>/dev/null | sed 's/^/    /'
}

case "$MODE" in
  oracle)    land_patch "$(need oracle.patch)"    "oracle.patch" ;;
  negative)  land_patch "$(need negative.patch)"  "negative.patch" ;;
  noop)      restore_baseline ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. Grade with:  KERNELBENCH_VERIFY_MODE=candidate bash /tests/test.sh   -> /logs/verifier/reward.json"
