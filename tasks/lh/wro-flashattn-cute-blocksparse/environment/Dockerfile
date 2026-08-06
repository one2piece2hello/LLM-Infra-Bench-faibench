# syntax=docker/dockerfile:1
# =============================================================================
# wro-flashattn-cute-blocksparse — OPEN-SOURCE, self-contained task image
#
#   Task: restore the FUSED CuTe block-sparse forward dispatch; the shipped tree runs an eager reference that materialises the O(seqlen^2) score matrix and applies a densified allow-mask.
#         Editable scope (the ONLY file the scope gate allows to change):
#     - flash_attn/cute/interface.py
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1, anchor calibrated on H20 / sm_90).
#   The CuTe DSL path targets Hopper (sm_90) kernels — on a non-Hopper card the fused path
#   may be unavailable, which is a HARDWARE limitation, not a packaging one.
#   Platform: linux/amd64 ONLY. No GPU needed to BUILD.
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/wro-flashattn-cute-blocksparse
#     docker build -f environment/Dockerfile -t fai/lh-flashattn-cute-blocksparse:oss .
#
# RUN     docker run --rm --gpus all -it fai/lh-flashattn-cute-blocksparse:oss
# SCORE   docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/lh-flashattn-cute-blocksparse:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = 189.21 is a FROZEN
#   constant read from /opt/verifier-correctness-manifest.json (baked below) and calibrated on
#   H20. The verifier supports mode dispatch natively, so re-measuring is ONE command (the
#   reference patch is passed BY PATH and is never baked into any layer):
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/lh-flashattn-cute-blocksparse:oss bash /tests/test.sh
#   Sanity: noop ≈ the no-op value; negative (KERNELBENCH_NEGATIVE_PATCH=...) MUST score 0.
#
# VERSION ANCHORS  (the CuTe DSL stack is version-brittle — these pins are load-bearing)
#     CUDA 12.6 · Python 3.10 · torch 2.7.0 (cu126) · cuda-python 12.9.7 · nvidia-cutlass-dsl 4.5.0
#   cuda-python is pinned to 12.9.7 on purpose: the CuTe DSL binds to its low-level bindings and
#   a different minor breaks the JIT at import time.
#
# HOW THE GRADED WORKING TREE IS REPRODUCED
#   /app/repo is the flash-attention source tree with the target CuTe dispatch CARVED OUT (an
#   eager reference in its place). It is VENDORED under environment/repo/ rather than cloned:
#   the shipped image built it from a pre-staged tarball, so no upstream commit was ever
#   recorded, and these bytes were RECOVERED FROM THE SHIPPED IMAGE AND VERIFIED — this
#   package's oracle.patch applies cleanly to them (`git apply --check`), which proves they are
#   the exact starting point every graded run saw.
#   THREE sibling tasks in this bench carve DIFFERENT functions of this SAME file, so the carve
#   is union-wide: the oracle-only marker `FlashAttnFunc.apply` occurs 0 times here and exactly
#   once after oracle.patch. The build asserts that. (Do NOT use the baseline's own class names
#   as a leak marker — they exist by construction and are DELETED by oracle.patch.)
#   /app/stubs/flash_attn_2_cuda.py is a stub for the legacy compiled extension so
#   `import flash_attn` works without building it; tests/test.sh puts it on PYTHONPATH.
#   environment/repo/ carries NO .git; the single baseline commit is created at build time.
#
#   oracle.patch / negative.patch are reviewer-only and excluded by the context-root .dockerignore.
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/wro-flashattn-cute-blocksparse-loop16:v1-ms1@sha256:f5b5174f8aa46f7b948232fce2a4050394dd534259e5942cb712988bd8065853
#   method: `crane manifest` -> the small crane-appended layers -> `crane blob` + tar,
#           overlaid in manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes,
#           and `git apply --check -p1 negative.patch` does too — i.e. this tree is the
#           exact carved baseline both reference patches were generated against.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG TORCH_VERSION=2.7.0
ARG CUDA_PYTHON_VERSION=12.9.7
ARG CUTLASS_DSL_VERSION=4.5.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}" \
 && python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- the CuTe DSL stack, exact pins. A constraint file keeps torch from being moved. ----
RUN printf 'torch==%s\n' "${TORCH_VERSION}" > /tmp/constraints.txt \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt \
        "cuda-python==${CUDA_PYTHON_VERSION}" "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__; import cuda; print('cuda-python OK')"

# ---- 🔴 NO pip-installed flash_attn twin: it would reinstate the reference implementation. ----
RUN python - <<'PY'
import importlib.util, sys
for name in ("flash_attn", "flash_attn_interface"):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f"BUILD_ASSERT_FATAL: {name} is installed in site-packages - that reinstates the reference implementation")
print("OK: no pip-installed flash_attn twin; /app/repo will be the only copy")
PY

RUN mkdir -p /app/repo /app/stubs /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY environment/stubs/ /app/stubs/
COPY instruction.md /app/instruction.md

# ---- baseline commit; core.fileMode=false is LOAD-BEARING (see the note below:
#      a filemode skew makes `git status --porcelain` report phantom out-of-scope paths in
#      EVERY mode and forces reward to 0). ----
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

COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
COPY environment/loop/ /opt/loop/
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh /app/workspace/run_dev_bench.sh \
 && if [ -f /opt/loop/submit.sh ]; then chmod 0755 /opt/loop/submit.sh; fi \
 && chown -R root:root /app/entrypoint.sh /app/timer.sh /opt/loop \
 && if [ -f /opt/loop/score_engine.sh ]; then chmod 0700 /opt/loop/score_engine.sh; fi \
 && if [ -f /opt/loop/sanitize_feedback.py ]; then chmod 0700 /opt/loop/sanitize_feedback.py; fi \
 && if [ -d /opt/loop/private ]; then chmod -R go-rwx /opt/loop/private; fi

# ---- the FROZEN anchor. tests/test.sh reads it from this exact path and falls back to 1.0
#      (every reward 0) if absent. This task's tests/ ships no manifest of its own, so the
#      calibrated constant recovered from the shipped image travels in environment/. ----
COPY environment/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

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
    n="$(grep -Fc 'FlashAttnFunc.apply' /app/repo/flash_attn/cute/interface.py || echo 0)"; \
    echo "[build] oracle-only marker 'FlashAttnFunc.apply' count in the carved scope file = $n (must be 0)"; \
    if [ "$n" != "0" ]; then echo "BUILD_ASSERT_FAIL: the carve is NOT intact - the fused dispatch is present"; rc=1; fi; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK / CARVE-INTACT ASSERTS PASS"

ENV PYTHONPATH=/app/repo:/app/stubs \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

# ---- build-time smoke.
#      HARD: the scope file and the stub are present; git base is one clean commit; anchor sane.
#      SOFT: `import flash_attn` — the CuTe DSL probes the device at import time, so on a
#            GPU-less build host this can fail for reasons unrelated to packaging. The
#            authoritative check is at runtime (docker run --gpus all ... python -c "import flash_attn").
RUN python - <<'PY'
import os, sys, json, subprocess
sys.path.insert(0, "/app/repo"); sys.path.insert(1, "/app/stubs")
import torch
print("torch", torch.__version__)
assert os.path.isfile("/app/repo/flash_attn/cute/interface.py")
assert os.path.isfile("/app/stubs/flash_attn_2_cuda.py"), "the compiled-extension stub is missing"
m = json.load(open("/opt/verifier-correctness-manifest.json"))
print("baked anchor ref_speedup =", m.get("ref_speedup"))
assert isinstance(m.get("ref_speedup"), (int, float)) and m["ref_speedup"] > 1, m
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],capture_output=True,text=True).stdout.strip()
assert n == "1", n
try:
    import flash_attn
    print("flash_attn imported from", flash_attn.__file__)
    assert flash_attn.__file__.startswith("/app/repo/"), flash_attn.__file__
    print("OK: flash_attn resolves from the carved tree at /app/repo")
except Exception as e:
    print("WARN: `import flash_attn` failed at BUILD time (the CuTe DSL usually needs a GPU):", repr(e))
    print("      verify at runtime: docker run --gpus all ... python -c \"import flash_attn; print(flash_attn.__file__)\"")
print("OK: scope file, stub, git base and anchor all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
