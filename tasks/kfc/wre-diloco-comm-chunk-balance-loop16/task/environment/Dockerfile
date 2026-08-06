# syntax=docker/dockerfile:1
# =============================================================================
# wre-diloco-comm-chunk-balance-loop16 — OPEN-SOURCE, self-contained task image
#
#   Task: balance the communication chunks of a low-communication (DiLoCo-style) training step, then make it as fast as possible while preserving observable behaviour.
#         Editable scope (the ONLY file the scope gate allows to change):
##     - submission/kernel.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   CPU ONLY — task.toml declares gpus = 0. The graded code and the whole verifier are
#   PURE PYTHON STDLIB (json/math/random/secrets/collections), so this image installs
#   NO torch and NO numpy: nothing in the measured path would use them.
#   Platform: any linux/amd64 or arm64 host (no compiled dependency at all).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# PROVENANCE of environment/repo/, environment/loop/, environment/runtime/ (independently re-derivable)
#   recovered from: <internal registry>/kernelbench/wre-diloco-comm-chunk-balance-loop16:v1@sha256:7c2e8cc4859b65d297c16c47608ffa78301ac080a1175d43721a6e39d00aa7b0
#   method: `crane manifest` -> the 7 layers under 20 KB at the end of the stack -> `crane blob`
#           + tar, overlaid in manifest order; `.git/` and the `.wh..wh..opq` opaque marker dropped.
#   verification: the harness scripts recovered this way (submit.sh / score_engine.sh /
#           sanitize_feedback.py / entrypoint.sh / timer.sh / run_dev_bench.sh) are BYTE-IDENTICAL
#           to the ones already published with the sibling CPU-lane tasks, and
#           environment/repo/submission/kernel.py is byte-identical to the starting stub kept
#           beside the original task package. This task ships no oracle.patch (its reference
#           exists only as a single-file variant, solution/kernel_oracle.py), so the tree is
#           additionally pinned by the image digest above.
##
# -----------------------------------------------------------------------------
# BUILD  (no network needed beyond the public base image — nothing is pip-installed)
#     cd <...>/wre-diloco-comm-chunk-balance-loop16/task
#     docker build -f environment/Dockerfile -t fai/kfc-wre-diloco-comm-chunk-balance-loop16:oss .
#
# RUN     docker run --rm -it fai/kfc-wre-diloco-comm-chunk-balance-loop16:oss
# SCORE   docker run --rm -v "$PWD/tests:/tests:ro" fai/kfc-wre-diloco-comm-chunk-balance-loop16:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative | baseline2)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = 1.66973 .
#   tests/test.sh resolves it from tests/ref_speedup.txt FIRST, then falls back to
#   /opt/verifier-correctness-manifest.json, then to 1.0 (which would make every reward 0).
#   The value was calibrated on the authoring CPU lane (Intel Sapphire Rapids).
#   This verifier dispatches modes by FILE (not by patch), and the reference is passed BY PATH
#   and never baked, so re-measuring is ONE command:
#     docker run --rm -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py \
#       fai/kfc-wre-diloco-comm-chunk-balance-loop16:oss bash /tests/test.sh
#   then write the measured speedup into tests/ref_speedup.txt. See tests/ref_speedup.caveat.md.
#   Sanity: noop (no override) ~= 1.0 -> gated to 0; negative MUST score 0.
#
# VERSION ANCHORS
#     Python 3.11 (stdlib only). The measured metric is `proxy_bottleneck_bytes`, a deterministic
#     counter rather than wall-clock, so the anchor is far less hardware-sensitive than a
#     timing anchor — but it is still tied to this interpreter version.
# =============================================================================
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- git is required: the scope gate runs `git -C /app/repo status --porcelain`. ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates procps coreutils tar gzip vim less file jq \
    && rm -rf /var/lib/apt/lists/*

# ---- venv at this EXACT path: tests/test.sh puts /opt/kernelbench-venv/bin at the head of PATH. ----
ENV VENV=/opt/kernelbench-venv
RUN python3 -m venv "$VENV"
ENV PATH=$VENV/bin:$PATH

# ---- the graded working tree. A REAL directory (not a symlink): the scope gate runs
#      `git -C /app/repo status --porcelain`, which cannot see through a symlink. ----
RUN mkdir -p /app/repo /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY instruction.md /app/instruction.md

# ---- the baseline commit = the diff base the scope gate compares against.
#      🔴 core.fileMode=false is LOAD-BEARING, not hygiene: a shipped image in this corpus had
#      core.fileMode=true while some files were 0775 on disk but 100644 in the index, so
#      `git status --porcelain` reported phantom out-of-scope paths in EVERY mode and forced
#      reward to 0. Setting it false at creation makes that class of defect impossible. ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && git init -q \
 && git config core.fileMode false \
 && printf '__pycache__/\n*.pyc\n' >> .gitignore \
 && git add -A \
 && git commit -qm "carved baseline (graded starting point)" \
 && rm -rf .git/logs \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && test "$(git config --get core.fileMode)" = "false" \
 && test -z "$(git status --porcelain)" \
 && echo "[build] /app/repo: 1 commit, no remote, fileMode=false, clean tree"

# ---- runtime contract (recovered verbatim from the shipped image) ----
COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh && chown root:root /app/entrypoint.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
RUN chmod 0755 /app/workspace/run_dev_bench.sh
COPY environment/loop/ /opt/loop/
RUN if [ -f /opt/loop/submit.sh ]; then chmod 0755 /opt/loop/submit.sh; fi \
 && chown -R root:root /opt/loop \
 && if [ -f /opt/loop/score_engine.sh ]; then chmod 0700 /opt/loop/score_engine.sh; fi \
 && if [ -f /opt/loop/sanitize_feedback.py ]; then chmod 0700 /opt/loop/sanitize_feedback.py; fi \
 && if [ -d /opt/loop/private ]; then chmod -R go-rwx /opt/loop/private; fi

# ---- the FROZEN anchor fallback. tests/test.sh prefers tests/ref_speedup.txt (mounted) and falls
#      back to THIS path; keeping both consistent makes the resolution explicit. Only the constant. ----
COPY environment/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + workload.py +
#   ref_speedup.txt) is MOUNTED FRESH at /tests when scoring. It ships in this package —
#   open-source, but outside the image so the graded harness cannot be edited from the session.

# ---- 🔴 ANTI-LEAK ASSERTS ----
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'baseline2*.patch' -o -name 'scope_card.json' -o -name 'kernel_oracle.py' \
              -o -name 'kernel_negative.py' -o -name 'kernel_baseline2.py' \
              -o -name 'degradation_manifest.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS"

ENV PYTHONPATH=/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app/repo

# ---- build-time smoke: the scope file is present and the graded module resolves FROM /app/repo
#      (tests/test.sh re-asserts the import origin at scoring time; this build-time probe uses
#      find_spec so it never EXECUTES the module, and it warns instead of failing when the spec
#      cannot be resolved — a hard failure here would be about the build host, not the package). ----
RUN python - <<'PY'
import os, subprocess, sys
sys.path.insert(0, "/app/repo")
for f in ('submission/kernel.py',):
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
import importlib.util
for _mod in ('submission.kernel',):
    try:
        _spec = importlib.util.find_spec(_mod)
    except Exception as _e:
        print("NOTE: import-origin probe skipped for", _mod, "->", repr(_e),
              "(tests/test.sh re-asserts this at scoring time)")
        continue
    _loc = (getattr(_spec, "origin", "") or "") if _spec else ""
    print("resolves:", _mod, "->", _loc or "<not found>")
    assert not _loc or _loc.startswith("/app/repo"), \
        f"{_mod} resolves OUTSIDE the graded tree: {_loc}"
assert os.path.isfile("/opt/loop/submit.sh"), "loop16 harness missing: /opt/loop/submit.sh"
assert os.path.isfile("/opt/verifier-correctness-manifest.json"), "anchor fallback missing"
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("OK: scope file, import origin, loop harness, anchor fallback and git base all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
