# syntax=docker/dockerfile:1
# =============================================================================
# s4-fft-longconv — OPEN-SOURCE, self-contained task image
#
#   Task: make the declared scope faster while preserving its observable behaviour exactly.
#         Editable scope (the ONLY files the scope gate allows to change):
##     (declared in tests/test.sh)
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1, anchor calibrated on H20); no GPU to BUILD.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wre-arch-s4-fft-longconv:v1@sha256:4215533296f1f970d093257f96b14d5cada8925b220966c41471b23d78c36542
#   method: `crane manifest` -> the layers under 80 MB -> `crane blob` + tar, overlaid in
#           manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: this task ships no oracle.patch (its reference exists only as a single-file variant), so the tree is pinned by the image digest above; the scope files and the import-origin gate are asserted at build.
##
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/s4-fft-longconv/task
#     docker build -f environment/Dockerfile -t fai/kfc-s4-fft-longconv:oss .
#
# RUN     docker run --rm --gpus all -it fai/kfc-s4-fft-longconv:oss
# SCORE   docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/kfc-s4-fft-longconv:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 NO ANCHOR TO RE-CALIBRATE (binary reward)
#   IMPLEMENTATION-CLASS TASK: reward is binary 0/1 (all correctness cases pass and no
#   cheating -> 1.0, else 0.0). There is NO ref_speedup anchor, so nothing to re-calibrate.
#   tests/test.sh scores correctness only; no ref_speedup.txt ships with this task and
#   none is read. Timing IS still recorded as diagnostic metadata (and the in-image
#   manifest keeps an oracle_ms record from the H20 validation run) but it never enters
#   the reward. The reference implementation is solution/kernel_oracle.py, applied by
#   copying it over the editable file; solution/kernel_negative.py is the known-bad
#   control that MUST score 0. See tests/ref_speedup.caveat.md.
#
# VERSION ANCHORS
#     CUDA 12.6 · Python 3.10 · torch 2.7.0 (cu126, brings its matching triton 3.3.x)
#   The original private base declared torch 2.5.1 / triton 3.1.0 and the task overlay then ran
#   `pip install torch==2.7.0` — 2.7.0 is therefore the version the anchor was calibrated under.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Compute capability you will RUN on (Triton JITs for the live device; this only affects
# hand-built CUDA extensions): 8.0 A100 / 8.9 L40S-4090 / 9.0 H100-H20 / 10.0 B200
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG TORCH_VERSION=2.7.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# ---- venv at this EXACT path: tests/test.sh puts /opt/kernelbench-venv/bin at the head of PATH,
#      and a bare python3 from anywhere else would not have torch. ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}" \
 && python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"

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
 && git commit -qm "baseline (graded starting point)" \
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

# ---- the FROZEN anchor fallback. tests/test.sh prefers tests/ref_speedup.txt (mounted) and falls
#      back to THIS path; keeping both consistent makes the resolution explicit. Only the constant. ----
COPY environment/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + the workload/harness
#   + ref_speedup.txt) is MOUNTED FRESH at /tests when scoring. It ships in this package —
#   open-source, but outside the image so the graded harness cannot be edited from the session.

# ---- 🔴 ANTI-LEAK ASSERTS (author-side artifact denylist) ----
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

ENV PYTHONPATH=/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

# ---- build-time smoke: the graded module imports FROM /app/repo (the verifier asserts the same
#      thing at scoring time via its import-origin gate) and the scope files are present. ----
RUN python - <<'PY'
import os, subprocess, sys
sys.path.insert(0, "/app/repo")
for f in ():
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
print("NOTE: no importable scope module for this task; tests/test.sh owns the\n       import-origin gate at scoring time")
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("OK: scope files, import origin and git base all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
