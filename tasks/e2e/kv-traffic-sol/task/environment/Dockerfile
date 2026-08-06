# syntax=docker/dockerfile:1
# =============================================================================
# e2e-b1-kv-traffic-sol — OPEN-SOURCE, self-contained task image
#
#   Task: minimise KV-cache traffic / layout cost. The agent edits
#         /app/repo/submission/kv_traffic.py (Triton kernel or a CUDA extension built with the
#         in-image nvcc are both fair game); the OPEN verifier under /tests times it and
#         scores log-speedup against a strong baseline.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Platform: linux/amd64 ONLY (the official vLLM wheel is manylinux x86_64).
#
# BUILD   (context = task/; no GPU needed — vLLM comes from its official prebuilt wheel)
#     cd <...>/kv-traffic-sol/task
#     docker build -f environment/Dockerfile -t fai/b1-kv-traffic-sol:oss .
#   Non-H20 GPUs: pass your compute capability so self-built CUDA extensions target it:
#     --build-arg TORCH_CUDA_ARCH_LIST=8.0   # A100   (8.9 L40S/4090 · 9.0 H100/H20 · 10.0 B200)
# RUN     (needs 1 NVIDIA GPU, driver >= 535)
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -it fai/b1-kv-traffic-sol:oss
# SCORE   (inside the container; tests/ is MOUNTED, never baked)
#     bash /tests/test.sh                        # -> /logs/verifier/reward.json
# RE-CALIBRATE the oracle anchor on YOUR hardware (ref_speedup is H20-calibrated!)
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/opt/negative:ro" \
#       -e VERIFIER_MODE=ceiling_triton_fused fai/b1-kv-traffic-sol:oss bash /tests/test.sh
#   then write the measured speedup into tests/verifier-correctness-manifest.json -> ref_speedup.value
#
# VERSION ANCHORS: CUDA 12.6 · Python 3.10 · torch 2.7.1 (cu126) · triton 3.3.1 · vllm 0.10.1.1
#   vLLM 0.10.1.1 hard-requires torch==2.7.1 — the constraint file pins torch/triton so vLLM's
#   resolver can never move them (its compiled _C.so must match the torch ABI).
#
# EDITABLE vLLM WORKING TREE: /app/repo/vllm is a SYMLINK to the installed package. Compiled
#   extensions find libtorch/nvidia libs via relative RPATHs ($ORIGIN/../torch/lib), so MOVING
#   the package out of site-packages would break dlopen; a symlink keeps the real files where
#   their RPATHs expect them while `import vllm` still resolves from /app/repo (PYTHONPATH).
#   One physical copy -> the agent's edits under /app/repo/vllm always take effect.
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

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

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.7.1" "triton==3.3.1"

# ---- vLLM from the OFFICIAL PUBLIC WHEEL, WITH its dependencies. constraint pins torch/triton
#      so vLLM's resolver cannot move them; pip still installs all declared deps (hand-listing
#      them is not viable — vllm declares ~68). No `|| true`: a failed install must fail the build.
RUN printf 'torch==2.7.1\ntriton==3.3.1\n' > /tmp/vllm-constraints.txt \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        --constraint /tmp/vllm-constraints.txt "vllm==0.10.1.1" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__; print('torch pin held:', torch.__version__)"

# ---- editable working tree at /app/repo (symlink; see header) ----
#      NOTE: no inline `#` comments inside this RUN — shell-form RUN joins the \-continued lines
#      into ONE command, where a `#` would comment out everything after it.
RUN mkdir -p /app/repo/submission /app/dev_bench /app/.cache/torch_extensions /logs/verifier /logs/artifacts \
 && VLOC="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" \
 && echo "vllm installed at $VLOC" \
 && ln -s "$VLOC" /app/repo/vllm \
 && test -f /app/repo/vllm/__init__.py
ENV PYTHONPATH=/app/repo \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64

# ---- solver-editable submission + public dev bench (from this task package) ----
COPY environment/submission/kv_traffic.py /app/repo/submission/kv_traffic.py
COPY environment/dev_bench/run_dev_bench.py /app/dev_bench/run_dev_bench.py
COPY environment/dev_bench/dev_suite.json /app/dev_bench/dev_suite.json
COPY instruction.md /app/instruction.md
RUN ln -sfn /app/repo/submission /app/submission

# ---- git baseline in /app/repo: ONE commit as the diff base (test.sh dumps git status/diff) ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && printf '__pycache__/\n*.pyc\n*.so.tmp\n.cache/\n' > .gitignore \
 && git init -q && git add -A && git commit -qm "baseline" \
 && test "$(git rev-list --count HEAD)" = "1" \
 && echo "git diff base ready: $(git rev-parse --short HEAD)"

# ---- OPEN grading suite is NOT baked. tests/ (test.sh + compute_reward.py + harness/* +
#      verifier-correctness-manifest.json, which carries the H20-calibrated ref_speedup used by
#      the reward formula) ships in this package and is MOUNTED at /tests when scoring, so the
#      solver cannot read or edit it while solving. Nothing may live under /opt/verifier.
RUN rm -rf /opt/verifier

# ---- runtime contract: entrypoint + elapsed-time daemon ----
RUN printf '#!/bin/bash\nset -e\nmkdir -p /app/.timer\nnohup /app/timer.sh >/app/.timer/timer.log 2>&1 &\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nmkdir -p /app/.timer\nSTART=$(date +%%s)\nwhile true; do echo $(( $(date +%%s) - START )) > /app/.timer/elapsed; sleep 5; done\n' > /app/timer.sh \
 && chmod 755 /app/timer.sh

ENV SUBMISSION_DIR=/app/repo/submission REPO_DIR=/app/repo
WORKDIR /app/repo/submission

# ---- build-time smoke. HARD: stack versions + tree layout + git base. SOFT: `import vllm`
#      (on a GPU-less build host vLLM's platform probe can fail for reasons unrelated to
#      packaging; the authoritative check is at runtime).
RUN python - <<'PY'
import os, glob, subprocess
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__)
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert triton.__version__.startswith("3.3"), triton.__version__
assert os.path.islink("/app/repo/vllm"), "/app/repo/vllm must be a symlink"
print("/app/repo/vllm ->", os.readlink("/app/repo/vllm"))
assert os.path.isfile("/app/repo/vllm/__init__.py")
print("compiled ext:", glob.glob("/app/repo/vllm/_C*.so") or "NONE (warn)")
assert os.path.isfile("/app/repo/submission/kv_traffic.py"), "submission entry missing"
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("git diff base OK")
try:
    import vllm
    print("vllm", vllm.__version__, "from", vllm.__file__)
    assert vllm.__file__.startswith("/app/repo/"), vllm.__file__
    print("OK: vllm imports from the editable tree at /app/repo")
except Exception as e:
    print("WARN: `import vllm` failed at BUILD time (often needs a GPU/driver):", repr(e))
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
