#!/bin/bash
# ============================================================================
# pre_artifacts.sh — kfc/wro-colossalai-devicemesh-rank-collate-loop16
#
# Run by the runner INSIDE this task's container AFTER the agent session ends and BEFORE
# grading (spec §1.1).  It turns whatever the agent left behind into a portable artifact:
#
#   /logs/artifacts/model.patch      unified diff: WORKING TREE vs the baked baseline commit
#   /logs/artifacts/model_files.txt  one changed/added ABSOLUTE path per line
#
# 🔴 IT NEVER COMMITS, and the task statement never asks the agent to (spec H3 — the one
#    place we deliberately differ from deep-swe, whose pre_artifacts.sh diffs <base>..HEAD).
#    Reason: grading reads the WORKING TREE.  `git status --porcelain` is the scope gate,
#    `git checkout -q HEAD -- <scope>` re-materialises the baseline for the paired timing,
#    and loop16's `submit.sh --finalize` plants the best iterate with read-tree +
#    checkout-index while HEAD stays on the baked baseline.  A commit moves HEAD, the scope
#    gate then sees nothing, `checkout HEAD -- scope` hands the CANDIDATE back as the
#    "baseline" — and a correct solve scores 0.
#    `git add -AN` (intent-to-add) is what makes NEW files appear in `git diff`: it stages
#    names only, writes no tree, creates no commit and leaves HEAD exactly where it was.
#
# EXIT CODE IS ALWAYS 0.  A missing or empty artifact must not fail the round; the verifier
# scores the container on its own terms and gives 0 by itself.
#
# READ-ONLY ON THE GRADED STATE.  `add -AN` is done against a COPY of the repo's index
# (GIT_INDEX_FILE), so after this script the repo's .git/index is byte-identical and the
# verifier — which runs after us — sees exactly the `git status --porcelain` it would have
# seen if this script had never run.  No commit, no HEAD move, no working-tree write.
#
# CAPTURE ROOTS for THIS task — derived from the package, never assumed:
#   /app/repo          git repo, HEAD == the 1-commit carved baseline
#       evidence  Dockerfile:144: WORKDIR /app/repo
#       evidence  Dockerfile:91: && cd /app/repo \
#       evidence  Dockerfile:98: && test "$(git rev-list --count HEAD)" = "1" \
#       evidence  test.sh:10: export PYTHONPATH="/app/repo:${PYTHONPATH:-}"
#       evidence  test.sh:13: REPO=/app/repo
#
# Single root => model.patch is a bare `git diff --binary <baseline>` (no banner).
# model_files.txt carries ABSOLUTE paths, corpus-wide, so the two forms stay parseable
# by one reader.
#
# loop16 task (environment/loop/ -> /opt/loop in the image): the harness auto-finalizes
# at k=16 and `submit.sh --finalize` plants the best iterate in the WORKING TREE while
# HEAD stays on the baked baseline — which is exactly what this script captures. Running
# it after a finalize therefore yields the GRADED artifact, not the last raw edit.
# ============================================================================
set -uo pipefail

OUT=/logs/artifacts
PATCH="$OUT/model.patch"
LIST="$OUT/model_files.txt"
MULTI=0                 # 1 => several capture roots => each diff gets a banner line
MAX_BLOB_KB=5120                # per-file cap for the tree-snapshot path (keeps the patch sane)

mkdir -p "$OUT" 2>/dev/null || true
: > "$PATCH" 2>/dev/null || true
: > "$LIST"  2>/dev/null || true
git config --global --add safe.directory '*' >/dev/null 2>&1 || true

banner() {                 # $1=root $2=mode ; only when there is more than one root
    if [ "$MULTI" = 1 ]; then
        printf '# ==== pre_artifacts root=%s mode=%s ====\n' "$1" "$2" >> "$PATCH" 2>/dev/null
    fi
    return 0
}

list_add() {               # stdin: paths relative to $1 -> absolute, one per line
    sed "s|^|${1%/}/|" >> "$LIST" 2>/dev/null || true
    return 0
}

capture_git() {            # $1 = dir whose HEAD is the baked 1-commit baseline
    root=$1
    if [ ! -d "$root" ]; then echo "[pre_artifacts] SKIP (absent): $root"; return 0; fi
    if ! git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "[pre_artifacts] WARN $root has no .git -> falling back to a tree snapshot"
        capture_tree "$root"
        return 0
    fi
    # 🔴 SCRATCH INDEX. `add -AN` has to touch an index, and the verifier runs AFTER us and
    # reads `git status --porcelain`; turning `?? newdir/` into `A newdir/file` there could
    # flip a scope gate either way. So copy the repo's index aside and stage into the COPY —
    # the repo's own .git/index is byte-identical when we are done.
    tmpi=$(mktemp -d /tmp/pre_artifacts.XXXXXX 2>/dev/null) || tmpi=""
    gd=$(git -C "$root" rev-parse --absolute-git-dir 2>/dev/null || true)
    if [ -z "$gd" ]; then
        gd=$(git -C "$root" rev-parse --git-dir 2>/dev/null || true)
        case "$gd" in /*) ;; ?*) gd="$root/$gd" ;; esac
    fi
    if [ -n "$tmpi" ] && [ -n "$gd" ] && [ -f "$gd/index" ] && cp "$gd/index" "$tmpi/index" 2>/dev/null; then
        GIT_INDEX_FILE="$tmpi/index"; export GIT_INDEX_FILE
    else
        echo "[pre_artifacts] WARN no scratch index for $root; staging into the real index"
    fi
    git -C "$root" add -AN . >/dev/null 2>&1 || true
    base=HEAD
    if ! git -C "$root" rev-parse -q --verify HEAD >/dev/null 2>&1; then
        base=$(git -C "$root" hash-object -w -t tree /dev/null 2>/dev/null)
        echo "[pre_artifacts] WARN $root has no commit; diffing against the empty tree"
    else
        n=$(git -C "$root" rev-list --count HEAD 2>/dev/null || echo 0)
        case "$n" in ''|*[!0-9]*) n=0 ;; esac
        if [ "$n" -gt 1 ]; then
            base=$(git -C "$root" rev-list --max-parents=0 HEAD 2>/dev/null | tail -1)
            echo "[pre_artifacts] 🔴 $root HEAD has $n commits but the contract says 1:"
            echo "[pre_artifacts]    the agent committed. Diffing from the ROOT commit ${base} so"
            echo "[pre_artifacts]    nothing is lost — but grading reads the WORKING TREE, so a moved"
            echo "[pre_artifacts]    HEAD can still zero an otherwise-correct solve."
        fi
    fi
    [ -n "${base:-}" ] || base=HEAD
    banner "$root" "git-diff-vs-baseline"
    git -C "$root" diff --binary "$base" >> "$PATCH" 2>/dev/null || true
    names=$(git -C "$root" diff --name-only "$base" 2>/dev/null)
    echo "[pre_artifacts] $root: baseline $(git -C "$root" rev-parse --short "$base" 2>/dev/null), $(printf '%s\n' "$names" | grep -c . ) changed path(s)"
    unset GIT_INDEX_FILE
    [ -n "$tmpi" ] && rm -rf "$tmpi" 2>/dev/null
    printf '%s\n' "$names" | grep -v '^$' | list_add "$root"
    return 0
}

capture_tree() {           # $1 = editable dir that is NOT a git repo -> full snapshot
    root=$1
    if [ ! -d "$root" ]; then echo "[pre_artifacts] SKIP (absent): $root"; return 0; fi
    tmp=$(mktemp -d /tmp/pre_artifacts.XXXXXX 2>/dev/null) || return 0
    if git init -q "$tmp/snap" >/dev/null 2>&1; then
        GD="$tmp/snap/.git"
        empty=$(git --git-dir="$GD" hash-object -w -t tree /dev/null 2>/dev/null)
        ( cd "$root" 2>/dev/null || exit 0
          find . -name .git -prune -o -name __pycache__ -prune -o \
                 -type f ! -name '*.pyc' ! -name '*.pyo' -size -"${MAX_BLOB_KB}"k -print0 \
            2>/dev/null \
          | GIT_DIR="$GD" GIT_WORK_TREE="$root" xargs -0 -r git add -f -- ) >/dev/null 2>&1 || true
        banner "$root" "tree-snapshot-not-a-git-repo"
        GIT_DIR="$GD" GIT_WORK_TREE="$root" git diff --cached --binary "$empty" \
            >> "$PATCH" 2>/dev/null || true
    else
        echo "[pre_artifacts] WARN could not stage a snapshot index for $root"
    fi
    rm -rf "$tmp" 2>/dev/null || true
    all=$( cd "$root" 2>/dev/null && find . -name .git -prune -o -name __pycache__ -prune -o \
              -type f -print 2>/dev/null | sed 's|^\./||' )
    printf '%s\n' "$all" | grep -v '^$' | list_add "$root"
    big=$( cd "$root" 2>/dev/null && find . -name .git -prune -o -type f \
              -size +"${MAX_BLOB_KB}"k -print 2>/dev/null | grep -c . )
    echo "[pre_artifacts] $root: tree snapshot (no git baseline here), $(printf '%s\n' "$all" | grep -c . ) file(s)"
    if [ "${big:-0}" != 0 ]; then
        echo "[pre_artifacts] NOTE $big file(s) under $root exceed ${MAX_BLOB_KB}KB: listed in"
        echo "[pre_artifacts]      model_files.txt but their bytes are NOT inlined into model.patch."
    fi
    return 0
}

capture_git "/app/repo"
bytes=$(wc -c < "$PATCH" 2>/dev/null || echo 0)
case "$bytes" in ''|*[!0-9]*) bytes=0 ;; esac
echo "[pre_artifacts] captured $bytes bytes, $(wc -l < "$LIST" 2>/dev/null || echo 0) path(s)"
if [ "$bytes" -gt 67108864 ]; then
    echo "[pre_artifacts] 🔴 NOTE model.patch is $bytes bytes (> 64MiB). Almost certainly a build"
    echo "[pre_artifacts]    product or a checkpoint was left inside a capture root — which is also"
    echo "[pre_artifacts]    an out-of-scope edit the verifier's scope gate will punish."
fi
exit 0
