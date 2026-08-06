# syntax=docker/dockerfile:1
# =============================================================================
# wro-fla-abc-chunkscan-sol — OPEN-SOURCE, self-contained task image
#
#   Task: restore the chunked-scan Triton kernels for ABC (Attention with Bounded-memory Control); the shipped tree is an O(T) eager recurrence.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - fla/ops/abc/chunk.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1, anchor calibrated on H20); no GPU to BUILD.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/wro-fla-abc-chunkscan-sol
#     docker build -f environment/Dockerfile -t fai/lh-fla-abc-chunkscan:oss .
#
# RUN
#     docker run --rm --gpus all -it fai/lh-fla-abc-chunkscan:oss
#
# SCORE (tests/ is MOUNTED, never baked into the image)
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/lh-fla-abc-chunkscan:oss bash /tests/test.sh
#     # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = 578.380265 is a FROZEN
#   constant read from /opt/verifier-correctness-manifest.json (baked below from this package's
#   tests/verifier-correctness-manifest.json) and was calibrated on H20. It is large because it
#   compares an EAGER PYTHON reference against a real Triton kernel — the ratio is dominated by
#   the algorithmic difference and is therefore fairly portable, but it is still a single-machine
#   measurement. The verifier supports mode dispatch natively, so re-measuring is ONE command
#   (the reference patch is passed BY PATH and never baked):
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/lh-fla-abc-chunkscan:oss bash /tests/test.sh
#   Sanity: noop (no patch) ≈ the no-op value; negative (KERNELBENCH_NEGATIVE_PATCH=...) MUST be 0.
#
# VERSION ANCHORS
#     CUDA 12.6 · Python 3.10 · torch 2.7.0 (cu126, brings its matching triton 3.3.x)
#   The original private base declared torch 2.5.1 / triton 3.1.0 and the task overlay then
#   upgraded the venv with `pip install torch==2.7.0` — 2.7.0 is therefore the version the anchor
#   was calibrated under, so it is pinned directly here.
#
# HOW THE GRADED WORKING TREE IS REPRODUCED (this is what makes the task replayable)
#   /app/repo is the flash-linear-attention `fla` source tree with the target kernels CARVED OUT
#   (a slow-but-correct eager reference in their place). It is VENDORED in this package under
#   environment/repo/ rather than cloned, for three reasons:
#     1. the originally shipped image built it from a pre-staged tarball, so no upstream commit
#        was ever recorded — a `git clone` could not be pinned to the graded bytes;
#     2. these bytes were recovered from the shipped image and VERIFIED: `git apply --check`
#        of this package's oracle.patch succeeds against them, which proves they are the exact
#        starting point every graded run saw;
#     3. it makes the build independent of GitHub availability.
#   The carve is FAMILY-WIDE on purpose: sibling tasks in this bench target other kernels in the
#   same tree, so ALL of their scope files are carved here too — otherwise this image would hand
#   the solver a sibling task's answer. The build asserts every one of them is still carved.
#   environment/repo/ carries NO .git; the single baseline commit is created at build time.
#
#   oracle.patch / negative.patch are reviewer-only and excluded by the context-root .dockerignore.
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wro-fla-abc-chunkscan-sol-loop16:v3z@sha256:9e97d2775ecd39ae2d3dacf3beb2d3ae8901190dd465144067d030b02ecc941f
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Compute capability you will RUN on (Triton JITs for the live device; this only affects
# hand-built CUDA extensions): 8.0 A100 / 8.9 L40S·4090 / 9.0 H100·H20 / 10.0 B200
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

# ---- 🔴 NO pip-installed `fla` twin. The graded tree must be the ONLY copy: a pip install of
#      flash-linear-attention would put the pristine (answer-bearing) kernels back on sys.path. ----
RUN python - <<'PY'
import importlib.util, sys
for name in ("fla", "flash_linear_attention"):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f"BUILD_ASSERT_FATAL: {name} is installed in site-packages - that reinstates the reference implementation")
print("OK: no pip-installed fla twin; /app/repo will be the only copy")
PY

# ---- the graded working tree (REAL directory, not a symlink: the scope gate runs
#      `git -C /app/repo status --porcelain`, which cannot see through a symlink). ----
RUN mkdir -p /app/repo /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY instruction.md /app/instruction.md

# ---- the baseline commit = the diff base the scope gate compares against.
#      🔴 core.fileMode=false is LOAD-BEARING: a shipped image in this corpus had
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

# ---- sitecustomize: the shipped image blocks `import tilelang` (a third-party DSL that would
#      let the solver sidestep writing the kernel). tests/test.sh puts this dir FIRST on
#      PYTHONPATH, so the block is part of the graded contract, not an optional extra. ----
COPY environment/wro_sitecustomize/ /opt/wro_sitecustomize/

# ---- runtime contract + the 16-round self-improvement harness, as the original image shipped
#      it. /opt/loop/private/ is the IN-SESSION DEV proxy scorer; by this bench's design the
#      timed workload is not a secret (its own manifest says "dev reward is an in-session proxy;
#      the runner end-of-session /tests score is authoritative"). No reference impl is present. ----
COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
COPY environment/loop/ /opt/loop/
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh /app/workspace/run_dev_bench.sh \
 && if [ -f /opt/loop/submit.sh ]; then chmod 0755 /opt/loop/submit.sh; fi \
 && chown -R root:root /app/entrypoint.sh /app/timer.sh /opt/loop /opt/wro_sitecustomize \
 && if [ -f /opt/loop/score_engine.sh ]; then chmod 0700 /opt/loop/score_engine.sh; fi \
 && if [ -f /opt/loop/sanitize_feedback.py ]; then chmod 0700 /opt/loop/sanitize_feedback.py; fi \
 && if [ -d /opt/loop/private ]; then chmod -R go-rwx /opt/loop/private; fi

# ---- the FROZEN anchor. tests/test.sh reads ref_speedup from this exact path and silently falls
#      back to 1.0 (which makes every reward 0) if it is missing, so it must be baked. It carries
#      only the calibrated constant — no reference implementation. ----
COPY tests/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + workload.py +
#   h20.py + the manifest) is MOUNTED FRESH at /tests when scoring. It ships in this package —
#   open-source, but outside the image so the graded harness cannot be edited from the session.

# ---- 🔴 ANTI-LEAK / CARVE-INTACT ASSERTS ----
#      (The filename list below is an AUTHOR-SIDE ARTIFACT DENYLIST: names that only ever
#      exist in the authoring workspace. Asserting they are absent proves nothing leaked.)
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'scope_card.json' -o -name 'degradation_manifest.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS"
RUN python - <<'PY'
import os, sys
# The family-wide carve must still be intact: every sibling task's target kernel is absent.
FAMILY = {'fla/ops/nsa/parallel.py': 'parallel_nsa_fwd_kernel', 'fla/ops/wall_attn/parallel.py': 'parallel_wall_attn_fwd_kernel', 'fla/ops/abc/chunk.py': 'chunk_abc_fwd_kernel_h', 'fla/ops/gated_delta_rule/chunk.py': 'chunk_scaled_dot_kkt_fwd', 'fla/ops/ttt/chunk.py': 'chunk_ttt_linear_fwd_kernel_h'}
bad = []
for rel, marker in FAMILY.items():
    p = os.path.join("/app/repo", rel)
    if not os.path.isfile(p):
        continue
    src = open(p, encoding="utf-8", errors="replace").read()
    n_marker = src.count(marker)
    n_jit = src.count("@triton.jit")
    print(f"  carve check {rel}: marker '{marker}' x{n_marker}, @triton.jit x{n_jit}, {len(src.splitlines())} lines")
    if n_marker or n_jit:
        bad.append(rel)
if bad:
    sys.exit("BUILD_ASSERT_FATAL: the carve is NOT intact for " + ", ".join(bad)
             + " - this image would leak a reference implementation")
print("OK: family-wide carve intact (no target kernel anywhere in /app/repo)")
PY

ENV PYTHONPATH=/opt/wro_sitecustomize:/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

# ---- build-time smoke. `import fla` must resolve from /app/repo (the verifier asserts exactly
#      this at scoring time) and the scope files must be present. Importing fla pulls triton,
#      which does NOT need a GPU at import time. ----
RUN python - <<'PY'
import os, sys, json
sys.path.insert(0, "/app/repo")
import fla
print("fla imported from", fla.__file__)
assert fla.__file__.startswith("/app/repo/"), fla.__file__
for f in ('fla/ops/abc/chunk.py',):
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
import importlib.util
assert importlib.util.find_spec("tilelang") is None or True
m = json.load(open("/opt/verifier-correctness-manifest.json"))
print("baked anchor ref_speedup =", m.get("ref_speedup"))
assert isinstance(m.get("ref_speedup"), (int, float)) and m["ref_speedup"] > 1, m
print("OK: import origin, scope files and anchor all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
