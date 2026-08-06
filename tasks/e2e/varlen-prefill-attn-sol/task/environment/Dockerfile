# syntax=docker/dockerfile:1
# =============================================================================
# e2e-d1-varlen-prefill-attn-sol — OPEN-SOURCE, self-contained task image
#
#   Task: optimise variable-length causal PREFILL attention. The agent edits
#         /app/repo/submission/varlen_prefill_attn.py (may write a Triton kernel or
#         compile a CUDA extension with the in-image nvcc); the OPEN verifier under
#         /tests times it and scores log-speedup vs a strong baseline.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Platform: linux/amd64 ONLY (the official vLLM wheel is manylinux x86_64).
#             On a non-x86 host add `--platform=linux/amd64` (runs under emulation).
#   Build context = the task/ directory (this Dockerfile lives in task/environment/).
#
# -----------------------------------------------------------------------------
# BUILD
#     cd <...>/varlen-prefill-attn-sol/task
#     docker build -f environment/Dockerfile -t fai/e2e-d1-varlen:oss .
#   No GPU is needed to build (vLLM comes from its official prebuilt wheel).
#   Non-H20 GPUs: pass your compute capability so self-built CUDA extensions target it:
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0 ...      # A100
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=8.9 ...      # L40S / 4090
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=9.0 ...      # H100 / H20 (default)
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=10.0 ...     # B200
#
# RUN  (needs 1 NVIDIA GPU, driver >= 535)
#     docker run --rm --gpus all \
#       -v "$PWD/tests:/tests:ro" -it fai/e2e-d1-varlen:oss
#
# SCORE (inside the container; tests/ is MOUNTED, never baked into the image)
#     bash /tests/test.sh                       # -> /logs/verifier/reward.json
#
# RE-CALIBRATE the oracle anchor on YOUR hardware (ref_speedup is H20-calibrated!)
#   The verifier resolves alternative implementations from $NEGATIVE_DIR, so mount
#   this package's solution/ there and run the ceiling mode:
#     docker run --rm --gpus all \
#       -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/opt/negative:ro" \
#       -e VERIFIER_MODE=ceiling_triton_prefill \
#       fai/e2e-d1-varlen:oss bash /tests/test.sh
#   Read the measured speedup from /logs/verifier/benchmark_results.json and write it
#   into tests/verifier-correctness-manifest.json -> ref_speedup.value before scoring
#   candidates on that hardware. (Modes: candidate | strong_baseline | ceiling_* | negative_*)
#
# VERSION ANCHORS (reproduce the original runtime; see task.toml)
#     CUDA 12.6 · Python 3.10 · torch 2.7.1 (cu126) · triton 3.3.1 · vllm 0.10.1.1
#   vLLM 0.10.1.1 *hard-requires* torch==2.7.1 — the constraint file below pins torch
#   and triton so vLLM's own resolver can never move them (ABI must match its _C.so).
#
# HOW THE EDITABLE vLLM WORKING TREE IS BUILT (and why import + edits both work)
#   1. pip installs the official vllm==0.10.1.1 wheel WITH its 68 declared deps
#      (constraint-pinned torch/triton) -> ships the ALREADY-COMPILED _C*.so, no GPU build.
#   2. /app/repo/vllm is a SYMLINK to that installed package. This is deliberate:
#      compiled extensions locate libtorch/nvidia libs through relative RPATHs
#      ($ORIGIN/../torch/lib, $ORIGIN/../../nvidia/*/lib), so physically MOVING the
#      package out of site-packages would break dlopen. A symlink keeps the real files
#      where their RPATHs expect them while making `import vllm` resolve from
#      /app/repo (PYTHONPATH) — which satisfies the verifier's import-origin check.
#      There is exactly ONE physical copy, so edits through /app/repo/vllm/**.py are
#      the real files and always take effect.
#      Trade-off (honest): because vllm is reached via a symlink, `git diff` inside
#      /app/repo tracks submission/ but not vLLM internals. This task only requires
#      editing submission/ (the instruction says vLLM's own code need not be used).
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- OS toolchain: compilers + nvcc (from the CUDA devel base) so the agent can build
#      its own Triton / CUDA kernels at solve time ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 clang cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git git-lfs curl wget ca-certificates procps psmisc coreutils \
        tar gzip unzip xz-utils vim less file jq tini \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# ---- venv. Keep this exact path: tests/test.sh pins PATH=/opt/kernelbench-venv/bin ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions
RUN pip install --no-cache-dir -U pip setuptools wheel

# ---- DL runtime stack from PUBLIC indexes, exact pins ----
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.7.1" "triton==3.3.1"

# ---- vLLM 0.10.1.1 from the OFFICIAL PUBLIC WHEEL, WITH its dependencies.
#      A constraint file pins torch/triton so vLLM's resolver cannot move them, while pip
#      still installs all 68 declared deps (fastapi, openai, pyzmq, xgrammar, numba, ray,
#      compressed-tensors, ... — `import vllm` needs them; hand-listing them is not viable).
#      NOTE: no `|| true` here on purpose — a failed dependency install must fail the build. ----
RUN printf 'torch==2.7.1\ntriton==3.3.1\n' > /tmp/vllm-constraints.txt \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        --constraint /tmp/vllm-constraints.txt "vllm==0.10.1.1" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__; print('torch pin held:', torch.__version__)"

# ---- Assemble the editable working tree at /app/repo.
#      /app/repo/vllm -> <site-packages>/vllm  (SYMLINK: keeps the compiled extension's
#      relative RPATHs valid; single physical copy; edits take effect immediately).
#      LD_LIBRARY_PATH additionally hard-wires torch's lib dir as insurance in case the
#      agent relocates or rebuilds parts of the tree.
#      NOTE: no inline `#` comments inside this RUN — shell-form RUN joins the \-continued
#      lines into ONE command, where a `#` would comment out everything after it. ----
RUN mkdir -p /app/repo /app/dev_bench /app/.cache/torch_extensions /logs/verifier /logs/artifacts \
 && VLOC="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" \
 && echo "vllm installed at $VLOC" \
 && ln -s "$VLOC" /app/repo/vllm \
 && test -f /app/repo/vllm/__init__.py \
 && ( ls /app/repo/vllm/_C*.so >/dev/null 2>&1 \
        && echo "compiled extension present: $(ls /app/repo/vllm/_C*.so)" \
        || echo "WARN: no _C*.so under the vllm package" )
ENV PYTHONPATH=/app/repo \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64

# ---- solver-editable submission skeleton + public dev bench (from this task package) ----
RUN mkdir -p /app/repo/submission
COPY environment/submission/varlen_prefill_attn.py /app/repo/submission/varlen_prefill_attn.py
COPY environment/dev_bench/ /app/dev_bench/
COPY instruction.md /app/instruction.md
RUN ln -sfn /app/repo/submission /app/submission

# ---- git baseline in /app/repo: ONE commit as the diff base, so `git diff` / harness
#      scope audits can see exactly what the agent changed (tests/test.sh dumps
#      `git status --porcelain` to /logs/artifacts/repo_status.txt when .git exists).
#      safe.directory is required because the container runs as root. ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && printf '__pycache__/\n*.pyc\n*.so.tmp\n.cache/\n' > .gitignore \
 && git init -q \
 && git add -A \
 && git commit -qm "baseline" \
 && test "$(git rev-list --count HEAD)" = "1" \
 && echo "git diff base ready: $(git rev-parse --short HEAD)"

# ---- OPEN grading suite is NOT baked into the image (by design).
#   The verifier surface (tests/: test.sh + compute_reward.py + harness/baseline_prefill.py +
#   bench_prefill.py + hidden_suite.json + verifier-correctness-manifest.json, which carries
#   the H20-calibrated oracle anchor ref_speedup=1.5317 used by the reward formula) is
#   MOUNTED FRESH at /tests when scoring. It ships in this package's tests/ directory —
#   open-source, but deliberately outside the image so the solver cannot read or edit it
#   while solving. See the RUN / SCORE / RE-CALIBRATE recipes in the header.

# ---- runtime contract: entrypoint + elapsed-time daemon (same as the original image) ----
RUN printf '#!/bin/bash\nset -e\nmkdir -p /app/.timer\nnohup /app/timer.sh >/app/.timer/timer.log 2>&1 &\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nmkdir -p /app/.timer\nSTART=$(date +%%s)\nwhile true; do echo $(( $(date +%%s) - START )) > /app/.timer/elapsed; sleep 5; done\n' > /app/timer.sh \
 && chmod 755 /app/timer.sh

ENV SUBMISSION_DIR=/app/repo/submission REPO_DIR=/app/repo
WORKDIR /app/repo/submission

# ---- build-time smoke.
#      HARD: torch/triton import, working-tree layout, git base, submission entry present.
#      SOFT: `import vllm` — on a GPU-less build host vLLM's platform probe can fail for
#            reasons unrelated to packaging, so it warns instead of breaking the build; the
#            authoritative check is at runtime (`docker run --gpus all ... python -c "import vllm"`).
RUN python - <<'PY'
import os, glob, subprocess, sys
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__)
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert triton.__version__.startswith("3.3"), triton.__version__

assert os.path.islink("/app/repo/vllm"), "/app/repo/vllm must be a symlink to the installed package"
print("/app/repo/vllm ->", os.readlink("/app/repo/vllm"))
assert os.path.isfile("/app/repo/vllm/__init__.py")
so = glob.glob("/app/repo/vllm/_C*.so")
print("compiled ext:", so or "NONE (warn)")
assert os.path.isfile("/app/repo/submission/varlen_prefill_attn.py"), "submission entry missing"
n = subprocess.run(["git", "-C", "/app/repo", "rev-list", "--count", "HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("git diff base OK")

try:
    import vllm
    print("vllm", vllm.__version__, "from", vllm.__file__)
    assert vllm.__file__.startswith("/app/repo/"), vllm.__file__
    assert vllm.__version__ == "0.10.1.1", vllm.__version__
    print("OK: vllm imports from the editable tree at /app/repo")
except Exception as e:
    print("WARN: `import vllm` failed at BUILD time (often needs a GPU/driver):", repr(e))
    print("      verify at runtime: docker run --gpus all ... python -c 'import vllm; print(vllm.__file__)'")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
