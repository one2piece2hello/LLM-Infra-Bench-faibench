# syntax=docker/dockerfile:1
# =============================================================================
# wre-verl-grpo-advantage-loop16 — OPEN-SOURCE, self-contained task image
#
#   Task: make the declared scope faster while preserving its observable behaviour exactly.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - workspace/submission/advantage_estimators.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   CPU ONLY — task.toml declares gpus = 0.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wre-verl-grpo-advantage-loop16:v1@sha256:b1ba41d9bcac3e1f1750e5a07948cdf753311f32c6bf6d494931d6fecbbe7fc8
#   method: `crane manifest` -> the layers under 80 MB -> `crane blob` + tar, overlaid in
#           manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: this task ships no oracle.patch (its reference exists only as a single-file variant), so the tree is pinned by the image digest above; the scope files and the import-origin gate are asserted at build.
##
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/wre-verl-grpo-advantage-loop16/task
#     docker build -f environment/Dockerfile -t fai/kfc-wre-verl-grpo-advantage-loop16:oss .
#
# RUN     docker run --rm -it fai/kfc-wre-verl-grpo-advantage-loop16:oss
# SCORE   docker run --rm -v "$PWD/tests:/tests:ro" fai/kfc-wre-verl-grpo-advantage-loop16:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = None .
#   tests/test.sh resolves it from tests/ref_speedup.txt FIRST, then falls back to
#   /opt/verifier-correctness-manifest.json, then to 1.0 (which would make every reward 0).
#   The value was calibrated on the authoring CPU lane (Intel Sapphire Rapids). The verifier supports mode dispatch natively, so
#   re-measuring is ONE command (the reference patch is passed BY PATH and never baked):
#     docker run --rm -v "$PWD/tests:/tests:ro" -v "<dir holding oracle.patch>:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/kfc-wre-verl-grpo-advantage-loop16:oss bash /tests/test.sh
#   then write the measured speedup into tests/ref_speedup.txt. See tests/ref_speedup.caveat.md.
#   Sanity: noop (no patch) ~= 1.0 -> gated to 0; negative MUST score 0.
#
# VERSION ANCHORS
#     Python 3.11 · torch 2.7.0 (CPU wheel)
#   The original private base declared torch 2.5.1 / triton 3.1.0 and the task overlay then ran
#   `pip install torch==2.7.0` — 2.7.0 is therefore the version the anchor was calibrated under.
# =============================================================================
FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.7.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && rm -rf /var/lib/apt/lists/*

# ---- venv at this EXACT path: tests/test.sh puts /opt/kernelbench-venv/bin at the head of PATH. ----
ENV VENV=/opt/kernelbench-venv
RUN python3 -m venv "$VENV"
ENV PATH=$VENV/bin:$PATH
RUN pip install --no-cache-dir -U pip setuptools wheel
# ---- CPU-ONLY torch on purpose: task.toml declares gpus = 0. The CPU wheel runs the SAME ATen
#      CPU kernels as the CUDA build, so numerics are unchanged, and the image is ~8 GB smaller. ----
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && python -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"
RUN pip install --no-cache-dir "numpy>=1.26,<3"

# ---- the graded working tree. A REAL directory (not a symlink): the scope gate runs
#      `git -C /app/repo status --porcelain`, which cannot see through a symlink. ----
RUN mkdir -p /app/repo /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
# ---- 🔴 the SCORED unit for this task is /app/workspace/submission/advantage_estimators.py
#      (tests/test.sh: SUB="$WORK/submission/advantage_estimators.py", WORK=/app/workspace).
#      Unlike the sibling kfc tasks this one is single-file-submission style: the graded file lives under
#      /app/workspace, NOT inside the /app/repo tree, so it needs its own COPY. ----
COPY environment/workspace/ /app/workspace/
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

# ---- runtime contract (copied verbatim from the shipped image) ----
COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh && chown root:root /app/entrypoint.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
RUN chmod 0755 /app/workspace/run_dev_bench.sh

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + the workload/harness
#   + ref_speedup.txt) is MOUNTED FRESH at /tests when scoring. It ships in this package —
#   open-source, but outside the image so the graded harness cannot be edited from the session.

# ---- 🔴 ANTI-LEAK ASSERTS ----
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'baseline2*.patch' -o -name 'scope_card.json' -o -name 'kernel_oracle.py' \
              -o -name 'degradation_manifest.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS"

ENV WRE_WORKSPACE=/app/workspace \
    PYTHONPATH=/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app/repo

# ---- build-time smoke: the graded module imports FROM /app/repo (the verifier asserts the same
#      thing at scoring time via its import-origin gate) and the scope files are present. ----
RUN python - <<'PY'
import os, subprocess, sys
sys.path.insert(0, "/app/repo")
# 🔴 This task is single-file-submission style: the SCORED unit lives under /app/workspace, NOT in the
#    /app/repo tree (tests/test.sh: WORK="${WRE_WORKSPACE:-/app/workspace}";
#    SUB="$WORK/submission/advantage_estimators.py"). Assert the real path.
SCORED = "/app/workspace/submission/advantage_estimators.py"
assert os.path.isfile(SCORED), f"the SCORED unit is missing: {SCORED}"
print("scored unit present:", SCORED, os.path.getsize(SCORED), "B")
assert os.path.isfile("/app/workspace/run_dev_bench.py"), "the public dev bench is missing"
print("dev bench present: /app/workspace/run_dev_bench.py")
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("OK: scope files, import origin and git base all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
