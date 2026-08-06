#!/usr/bin/env bash
# =============================================================================
# solve.sh -- e2e/a8-peft-adapter-byte-golf
#   PEFT adapter byte-golf: best held-out CE per adapter byte (H20)
#
# ORACLE FORM      : DIRECTORY + TRAINING RUN. The reference is a recipe that must be EXECUTED:
#                    it writes $SUBMISSION_DIR/adapter.bin and copies its own adapter_entry.py.
#                    Nothing can be pre-baked, because the graded artifact is trained bytes.
# EDITABLE SURFACE : $SUBMISSION_DIR (default /app/submission)
# GRADED UNIT      : /app/submission/adapter_entry.py (needs build_adapted_model) +
#                    /app/submission/adapter.bin (byte-budget-capped, dual-measured)
# PRIMARY METRIC   : adaptation_gain_ratio (heldout CE) -> min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
# MODE VARIABLE    : VERIFIER_MODE (candidate|strong_baseline|ceiling|naive|negative_overbudget|
#                    negative_degenerate|negative_stash). In every non-candidate mode tests/test.sh
#                    does the seeding AND the training itself from an UPLOADED seed dir
#                    (CEILING_DIR / STRONG_BASELINE_DIR / NEGATIVE_DIR / STASH_ORACLE_DIR).
#                    solve.sh reproduces the same seeding locally so you can grade in candidate mode;
#                    the equivalent reviewer command is printed at the end of every run.
# REFERENCE ASSETS : solution/{ceiling,strong_baseline}/train_adapter.py, solution/ceiling/adapter_entry.py,
#                    solution/naive/{train_adapter.py,adapter_entry_degenerate.py},
#                    solution/stash_oracle/adapter_entry.py
# SHA-PINNED PATHS : tests/verifier-correctness-manifest.json: compute_reward_sha256, test_sh_sha256,
#                    thresholds.base_model_sha256 (/app/base_model/model.safetensors), held_out.sha256,
#                    data_loader_checksum_G6.{train,val,tokenizer}_sha256, frozen_surface =
#                    /tests/* + /app/base_model. solve.sh writes ONLY under $SUBMISSION_DIR and never
#                    reads or writes /app/base_model or /tests.
#
# GRADE AFTER LANDING (one command, run from the task/ dir of the package):
#   docker run --rm --gpus all -v "$PWD:/task:ro" -v "$PWD/tests:/tests:ro" \
#     fai/e2e-a8-peft-adapter-byte-golf:oss bash -lc '
#       bash /task/solution/solve.sh && bash /tests/test.sh'
#   # (SOLVE_SKIP_TRAIN=1 stages the files but skips the ~1400-step training run)
#   -> /logs/verifier/reward.json
#
# CLI: (default)=oracle | --negative | --baseline2 | --noop | --variant <name> | --list | --help
#
# --noop SEMANTICS : --noop 把可编辑面复位到烤入基线（与 verifier 取基线的方式一致，HEAD 不动），
#                    使对照组能量到真正的 no-op 值。
#                    (`--noop` resets the editable surface to the baked baseline -- the same way the verifier
#                    materialises its own timing/scoring baseline -- with HEAD untouched, so the control arm
#                    measures the REAL no-op value instead of whatever a previous arm left behind.)
#                    HOW, here: rm -f $SUB/adapter.bin and restore adapter_entry.py from
#                    the pre-landing snapshot under ${SOLVE_BACKUP_DIR:-/tmp/.fai_solve_baseline}/<task-id>/
#                    BUT NOTE: the pristine image ships NO adapter.bin at all, so the LITERAL no-op arm
#                    hard-fails the entry contract (reward 0, build_or_entry_contract_failed). The
#                    model-visible FLOOR arm is `--variant naive` (equivalently -e VERIFIER_MODE=naive,
#                    which makes test.sh train the baked starter itself). --noop prints this too.
#
# HEAD MUST NOT MOVE. solve.sh only writes files into the editable surface -- no git add,
# no commit, no branch. The verifier grades the WORK TREE and (where that surface is a git
# tree) gates on `git status --porcelain`, so a commit turns a correct solution into
# reward 0 (RUNNABLE_SPEC H3). HEAD is re-read after landing; the script aborts if it moved.
#
# NOTE: this is the ONE task whose landing has to RUN a training job (a8 grades trained bytes,
#       so there is nothing to copy). Expect a long GPU run. SOLVE_SKIP_TRAIN=1 stages the entry
#       module only -- the entry contract then fails on the missing adapter.bin, by design.
#       solution/strong_baseline/ deliberately ships NO adapter_entry.py: its train_adapter.py
#       reuses whatever entry module is already in the submission dir (the baked starter).
# =============================================================================
set -uo pipefail

TASK_ID='a8-peft-adapter-byte-golf'
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
  printf '  %-26s %s\n' '(default) / --oracle' 'ceiling -- int4 nibble-packed LoRA @1400 steps (the ref_speedup anchor)'
  printf '  %-26s %s\n' '--baseline2' 'strong_baseline -- the 1.0 anchor recipe (keeps the baked entry module)'
  printf '  %-26s %s\n' '--negative' 'naive/adapter_entry_degenerate.py + a 1 KiB zero adapter.bin (must score 0)'
  printf '  %-26s %s\n' '--variant naive' 'the model-visible naive recipe (trains it)'
  printf '  %-26s %s\n' '--variant stash' 'stash_oracle entry + a 4 KiB adapter.bin (hidden-stash negative)'
  printf '  %-26s %s\n' '--noop' 'clear adapter.bin + reset the entry; see --noop SEMANTICS in the header'
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

train_recipe() { # train_recipe <variant-dir-relpath>
  local dir script
  script="$(need "$1/train_adapter.py")"; dir="$(dirname "$script")"
  if [ -n "${SOLVE_SKIP_TRAIN:-}" ]; then
    warn "SOLVE_SKIP_TRAIN=1 -> NOT running $script. \$SUB/adapter.bin will be missing and the"
    warn "entry contract will hard-fail (build_or_entry_contract_failed). Unset it to really land."
    return 0
  fi
  say "running the reference recipe (this trains -- expect a long GPU run):"
  say "  cd $SUB && python3 $script"
  snap "$SUB/adapter.bin"; snap "$SUB/adapter_entry.py"
  ( cd "$SUB" && SUBMISSION_DIR="$SUB" python3 "$script" ) \
    || die "reference training failed ($script). Re-run with the output above for the cause."
  [ -s "$SUB/adapter.bin" ] || die "training finished but $SUB/adapter.bin is missing/empty"
  say "trained artifact: $(wc -c < "$SUB/adapter.bin") bytes at $SUB/adapter.bin"
  say "entry module:     $SUB/adapter_entry.py"
}

case "$MODE" in
  oracle)    land 'solution/ceiling/adapter_entry.py=>adapter_entry.py'
             train_recipe solution/ceiling ;;
  baseline2) train_recipe solution/strong_baseline ;;
  naive)     land 'solution/naive/adapter_entry_degenerate.py=>adapter_entry.py'
             train_recipe solution/naive ;;
  negative)  land 'solution/naive/adapter_entry_degenerate.py=>adapter_entry.py'
             snap "$SUB/adapter.bin"
             head -c 1024 /dev/zero > "$SUB/adapter.bin" \
               && say "seeded a 1 KiB zero adapter.bin (degenerate negative; expect reward 0)" ;;
  stash)     land 'solution/stash_oracle/adapter_entry.py=>adapter_entry.py'
             snap "$SUB/adapter.bin"
             head -c 4096 /dev/urandom > "$SUB/adapter.bin" \
               && say "seeded a 4 KiB random adapter.bin (stash negative; expect reward 0)" ;;
  noop)      snap "$SUB/adapter.bin"; rm -f "$SUB/adapter.bin"
             restore adapter_entry.py
             say "the pristine image ships NO adapter.bin, so the literal no-op arm hard-fails the"
             say "entry contract. The model-visible FLOOR arm is: -e VERIFIER_MODE=naive (test.sh"
             say "trains the baked starter itself), or: bash solve.sh --variant naive" ;;
  *) die "no such variant: $MODE (see --list)" ;;
esac
say "done. GRADE WITH:"
say '  bash /tests/test.sh          # candidate mode; the trained adapter.bin is the graded artifact'
say '  reviewer-mode equivalent (test.sh trains the recipe itself):'
say '    docker run ... -v <pkg>/solution/ceiling:/opt/ceiling:ro -e VERIFIER_MODE=ceiling -e CEILING_DIR=/opt/ceiling <image> bash /tests/test.sh'
