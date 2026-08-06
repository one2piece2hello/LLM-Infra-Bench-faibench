# syntax=docker/dockerfile:1
# =============================================================================
# e2e-a8-peft-adapter-byte-golf — OPEN-SOURCE, self-contained task image
#
#   Task: adapt a FROZEN Qwen2.5-0.5B-Instruct to a new domain while shipping at most
#         327 680 bytes of adapter artifact. The agent writes
#         /app/submission/{train_adapter.py,adapter_entry.py} and may study/adapt the
#         full 🤗 PEFT source at /app/repo. Reward = bounded log of the adaptation-gain
#         ratio over a strong baseline, on a held-out split it never sees.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = the task/ directory (this Dockerfile lives in task/environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network: PyPI + github.com + huggingface.co)
#     cd <...>/a8-peft-adapter-byte-golf/task
#     docker build -f environment/Dockerfile -t fai/e2e-a8-peft-adapter-byte-golf:oss .
#   No GPU needed to build. Non-H20 GPUs: set your compute capability:
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0 ...   # A100  (8.9 L40S/4090, 9.0 H100/H20)
#
# RUN  (needs 1 NVIDIA GPU with bf16, driver >= 535)
#     docker run --rm --gpus all --shm-size=8g -it fai/e2e-a8-peft-adapter-byte-golf:oss
#
# SCORE (inside the container; tests/ is MOUNTED, never baked into the image)
#     docker run --rm --gpus all --shm-size=8g \
#       -v "$PWD/tests:/tests:ro" fai/e2e-a8-peft-adapter-byte-golf:oss bash /tests/test.sh
#     # -> /logs/verifier/reward.json
#
# 🔴 RE-CALIBRATE THE ANCHORS ON YOUR BUILD  (read this before trusting a reward)
#   reward = min(1.0, ln(gain_ratio/ref_speedup)/ln(ref_speedup)) if gain_ratio > ref_speedup else 0.0, with the FROZEN constants in
#   tests/verifier-correctness-manifest.json:
#       thresholds.strong_baseline_ce = 1.454941697688918   <- corpus-dependent!
#       ref_speedup                   = 1.2326578617539896  <- corpus-dependent!
#       thresholds.expected_base_ce   = 1.5657425161470404  (re-measured LIVE every run,
#                                        drift tol 0.05 — depends only on the frozen base
#                                        weights + the held-out corpus, both byte-pinned,
#                                        so this one is portable)
#   Both corpus-dependent constants were calibrated on H20 on the ORIGINAL /data/datasets/a8_syscode
#   corpus, which the task package does not ship (see "THE TRAINING CORPUS" below). The build
#   compares your regenerated corpus against the recorded digests and WARNS LOUDLY on mismatch.
#   On mismatch (or on a different GPU), re-calibrate with the SHIPPED reviewer recipes — this
#   task's verifier DOES honour mode dispatch:
#     # strong baseline -> new strong_baseline_ce
#     docker run --rm --gpus all --shm-size=8g -v "$PWD/tests:/tests:ro" \
#       -v "$PWD/solution/strong_baseline:/opt/strong_baseline:ro" \
#       -e VERIFIER_MODE=strong_baseline -e STRONG_BASELINE_DIR=/opt/strong_baseline \
#       fai/e2e-a8-peft-adapter-byte-golf:oss bash /tests/test.sh
#     # ceiling -> new ref_speedup = ceiling_ce_gain / strong_baseline_ce_gain
#     docker run --rm --gpus all --shm-size=8g -v "$PWD/tests:/tests:ro" \
#       -v "$PWD/solution/ceiling:/opt/ceiling:ro" -e VERIFIER_MODE=ceiling \
#       fai/e2e-a8-peft-adapter-byte-golf:oss bash /tests/test.sh
#   Read the CE / gain from /logs/verifier/benchmark_results.json (median of >=3 runs) and write
#   the new values into tests/verifier-correctness-manifest.json.
#   (Modes: candidate | strong_baseline | ceiling | naive | negative_*. gain_ratio <= 1 or an
#    invalid ref_speedup => reward 0 with a NAMED reason, by design.)
#
# VERSION ANCHORS
#     CUDA 12.6 · Python 3.10 · torch 2.7.0 (cu126) · transformers + accelerate + safetensors
#     🤗 PEFT source @ e4fe61b (the commit recorded in the manifest as source_commit_app_repo)
#     frozen base: Qwen/Qwen2.5-0.5B-Instruct, bf16 (Apache-2.0, ~1 GB)
#
# THE FROZEN BASE MODEL IS DIGEST-VERIFIED
#   The verifier HARD-CHECKS sha256(/app/base_model/model.safetensors) against
#   thresholds.base_model_sha256 = fdf756fa...288fb7fe. This build downloads the weights from the
#   public HF repo and asserts that digest at BUILD time, so a wrong revision fails here rather
#   than as a mysterious zero at scoring. Pin a different revision with --build-arg BASE_MODEL_REV=.
#
# THE TRAINING CORPUS IS REGENERATED, NOT SHIPPED  (⚠️ the one honest gap in this port)
#   /data/datasets/a8_syscode/{train,val}.txt was "real distributed-training systems Python
#   source". The task package ships neither the files nor the script that assembled them, and the
#   manifest records only their digests — so the ORIGINAL bytes are NOT recoverable. This build
#   reconstructs an equivalent corpus deterministically from PINNED PUBLIC repositories
#   (DeepSpeed / Megatron-LM / gpt-neox / torchtitan: every *.py, sorted by path, fixed byte
#   budget), then compares the digests and prints a prominent WARNING if they differ.
#   What this does and does not affect:
#     * the GRADED held-out split ships in tests/ (holdout_corpus.txt, digest-pinned) — unaffected;
#     * expected_base_ce is re-measured live on that held-out split — unaffected;
#     * strong_baseline_ce and ref_speedup were measured after training ON THIS CORPUS — these
#       two need re-calibration if the digests differ (recipe above).
#
# THE PRISTINE-IMAGE INVENTORY
#   The byte cap is enforced by diffing the filesystem against an inventory of the pristine image
#   (compute_reward.py FAILS CLOSED without it). The original baked it into /tests; since /tests
#   is now mounted read-only at scoring, this build writes it to /opt/a8_inventory/ in the FINAL
#   layer and points the verifier at it via E2E_INVENTORY_PATH (the escape hatch the verifier
#   itself provides: _frozen(name, env) prefers the env path).
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG PEFT_REF=e4fe61b43151da1125b8d6e6846250f66150e58b
ARG BASE_MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct
ARG BASE_MODEL_REV=7ae557604adf67be50417f59c2c2f167def9a775
ARG BASE_MODEL_SHA256=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
ARG CORPUS_TRAIN_BYTES=8388608
ARG CORPUS_VAL_BYTES=1048576
ARG DEEPSPEED_REF=v0.15.4
ARG MEGATRON_REF=core_r0.9.0
ARG NEOX_REF=v2.0
ARG TORCHTITAN_REF=v0.1.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git git-lfs curl wget ca-certificates procps psmisc coreutils \
        tar gzip unzip xz-utils vim less file jq tini \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# ---- venv. Keep this exact path: tests/test.sh puts /opt/kernelbench-venv/bin at the head of
#      PATH, and a bare `python3` from anywhere else would not have torch. ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.7.0" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__; print('torch', torch.__version__)"
RUN printf 'torch==2.7.0\n' > /tmp/constraints.txt \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt \
        "transformers>=4.44,<5" "accelerate>=0.33,<2" "safetensors>=0.4,<1" \
        "numpy>=1.26,<3" "sentencepiece==0.2.1" "huggingface_hub[cli]>=0.24,<1" \
 && python -c "import torch, transformers, accelerate, safetensors; print('transformers', transformers.__version__); assert torch.__version__.startswith('2.7.0')"

# ---- /app/repo: the 🤗 PEFT source at the recorded commit, fully editable, history stripped.
#      Fetched BY SHA so the tree is exactly the one recorded in the manifest. NOT pip-installed
#      and NOT on PYTHONPATH — same as the original image: `import peft` is something the solver
#      wires up deliberately (the verifier runs with PYTHONPATH unset). ----
RUN set -eu; \
    mkdir -p /app/repo; cd /app/repo; \
    git init -q; \
    git remote add origin https://github.com/huggingface/peft.git; \
    git fetch -q --depth 1 origin "${PEFT_REF}"; \
    git checkout -q FETCH_HEAD; \
    rm -rf /app/repo/.github /app/repo/docs/build; \
    test -f /app/repo/src/peft/__init__.py || { echo "BUILD_ASSERT_FATAL: peft source missing at /app/repo/src/peft"; exit 1; }; \
    n="$(find /app/repo -type f -not -path '*/.git/*' | wc -l)"; \
    echo "[build] /app/repo file count = $n"; \
    test "$n" -ge 100 || { echo "BUILD_ASSERT_FATAL: /app/repo has only $n files - clone mutilated"; exit 1; }
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && rm -rf .git \
 && printf '__pycache__/\n*.pyc\n' > .gitignore \
 && git init -q \
 && git add -A \
 && git commit -qm "huggingface/peft @ pinned commit (upstream history stripped)" \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && echo "git diff base ready: $(git rev-parse --short HEAD)"

# ---- /opt/peft_ro/src: a root-owned READ-ONLY copy that stays importable during the eval window.
#      The solver's /app/repo is quarantined at scoring, so the eval must not depend on it. ----
RUN mkdir -p /opt/peft_ro \
 && cp -r /app/repo/src /opt/peft_ro/src \
 && chown -R root:root /opt/peft_ro \
 && chmod -R a-w /opt/peft_ro \
 && test -f /opt/peft_ro/src/peft/__init__.py

# ---- the FROZEN base model. Digest-asserted at build time against the manifest's hard check. ----
RUN set -eu; \
    huggingface-cli download "${BASE_MODEL_REPO}" --revision "${BASE_MODEL_REV}" \
        --local-dir /app/base_model \
        --exclude "onnx/*" "*.onnx" "original/*" "*.gguf"; \
    test -f /app/base_model/model.safetensors || { echo "BUILD_ASSERT_FATAL: model.safetensors missing"; exit 1; }; \
    got="$(sha256sum /app/base_model/model.safetensors | awk '{print $1}')"; \
    echo "[build] base model.safetensors sha256 = $got"; \
    if [ "$got" != "${BASE_MODEL_SHA256}" ]; then \
      echo "BUILD_ASSERT_FATAL: frozen base weights do NOT match the digest the verifier hard-checks."; \
      echo "  expected ${BASE_MODEL_SHA256}"; \
      echo "  got      $got"; \
      echo "  Fix: pass --build-arg BASE_MODEL_REV=<the revision that matches>, or update"; \
      echo "       tests/verifier-correctness-manifest.json -> thresholds.base_model_sha256 AND"; \
      echo "       re-calibrate expected_base_ce / strong_baseline_ce / ref_speedup (see header)."; \
      exit 1; \
    fi; \
    echo "OK: frozen base weights match thresholds.base_model_sha256"; \
    python -c "import json;c=json.load(open('/app/base_model/config.json'));print('arch',c['architectures'],'dtype',c.get('torch_dtype'))"; \
    du -sh /app/base_model
RUN chmod -R a-w /app/base_model

# ---- the VISIBLE target-domain corpus, reconstructed from PINNED PUBLIC repositories.
#      Deterministic: fixed refs, every *.py sorted by relative path, fixed byte budgets.
#      The recorded digests are compared and a MISMATCH WARNS LOUDLY (it does not fail the build:
#      the original bytes are unrecoverable, and the task is still fully playable — only the two
#      corpus-dependent anchors need re-calibration). ----
RUN set -eu; mkdir -p /tmp/a8src /data/datasets/a8_syscode; cd /tmp/a8src; \
    git clone -q --depth 1 --branch "${DEEPSPEED_REF}"  https://github.com/microsoft/DeepSpeed.git   deepspeed; \
    git clone -q --depth 1 --branch "${MEGATRON_REF}"   https://github.com/NVIDIA/Megatron-LM.git    megatron; \
    git clone -q --depth 1 --branch "${NEOX_REF}"       https://github.com/EleutherAI/gpt-neox.git   gptneox; \
    git clone -q --depth 1 --branch "${TORCHTITAN_REF}" https://github.com/pytorch/torchtitan.git    torchtitan; \
    echo "[build] cloned 4 pinned distributed-training repos"
RUN python - <<'PY'
import hashlib, os, pathlib, sys

SRC = pathlib.Path("/tmp/a8src")
OUT = pathlib.Path("/data/datasets/a8_syscode")
TRAIN_BYTES = int(os.environ.get("CORPUS_TRAIN_BYTES", 8 << 20))
VAL_BYTES = int(os.environ.get("CORPUS_VAL_BYTES", 1 << 20))
RECORDED = {
    "train.txt": "255d7c2c4a783ec61f67b02e252ff00eb4b48f5c547440ad26f609c52219c3be",
    "val.txt": "a2462095761c108a2b34d4a9a314c34772fe666052220552a5bda1c5d24d74cb",
}

files = []
for repo in sorted(p.name for p in SRC.iterdir() if p.is_dir()):
    root = SRC / repo
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        if "/.git/" in "/" + rel or "/test" in rel or rel.endswith("_test.py"):
            continue
        files.append((rel, f))
print(f"[corpus] candidate .py files: {len(files)}")
if len(files) < 500:
    sys.exit(f"BUILD_ASSERT_FATAL: only {len(files)} python files collected - a clone is mutilated")

def emit(handle, budget, cursor):
    written = 0
    while cursor < len(files) and written < budget:
        rel, path = files[cursor]
        cursor += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        chunk = f"# ==== {rel} ====\n{text}\n"
        handle.write(chunk)
        written += len(chunk.encode("utf-8"))
    return written, cursor

cursor = 0
with open(OUT / "train.txt", "w", encoding="utf-8") as fh:
    n_train, cursor = emit(fh, TRAIN_BYTES, cursor)
with open(OUT / "val.txt", "w", encoding="utf-8") as fh:
    n_val, cursor = emit(fh, VAL_BYTES, cursor)
print(f"[corpus] train.txt {n_train:,} bytes | val.txt {n_val:,} bytes (disjoint file sets)")
if n_train < TRAIN_BYTES // 2 or n_val < VAL_BYTES // 2:
    sys.exit("BUILD_ASSERT_FATAL: corpus far below the requested byte budget")

drift = False
for name, want in RECORDED.items():
    got = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    print(f"[corpus] {name} sha256 = {got}")
    if got != want:
        drift = True
if drift:
    print("=" * 78)
    print("!!! WARNING: the regenerated corpus differs from the one the anchors were calibrated on.")
    print("!!! The ORIGINAL /data/datasets/a8_syscode bytes are not shipped and not recoverable.")
    print("!!! The task is fully playable, but these two constants in")
    print("!!!   tests/verifier-correctness-manifest.json are now STALE:")
    print("!!!     thresholds.strong_baseline_ce = 1.454941697688918")
    print("!!!     ref_speedup                   = 1.2326578617539896")
    print("!!! Re-calibrate with the shipped reviewer recipes (solution/strong_baseline and")
    print("!!! solution/ceiling) before comparing rewards - see this Dockerfile's header.")
    print("!!! expected_base_ce is re-measured live on the digest-pinned held-out split and is")
    print("!!! NOT affected.")
    print("=" * 78)
else:
    print("OK: regenerated corpus is byte-identical to the calibrated one")
PY
RUN rm -rf /tmp/a8src && chmod -R a-w /data/datasets/a8_syscode

# ---- solver-owned submission: the working-but-unambitious STARTER (not the answer) ----
RUN rm -rf /app/submission && mkdir -p /app/submission /logs/verifier /logs/artifacts
COPY environment/submission/ /app/submission/
COPY instruction.md /app/instruction.md

# ---- OPEN grading suite is NOT baked into the image (by design).
#   tests/ (test.sh + compute_reward.py + holdout_corpus.txt + the frozen manifest) is MOUNTED
#   FRESH at /tests when scoring. The original overlay baked BOTH /tests AND the reviewer-only
#   /opt/stash_oracle + /opt/strong_baseline + /opt/negative into the image behind root-0700 —
#   which is no protection at all when the container runs as uid 0. This build ships none of them;
#   the reviewer recipes live in this package's solution/ and are MOUNTED only when re-calibrating.

# ---- 🔴 ANTI-LEAK ASSERTS ----
RUN set -eu; rc=0; \
    for p in /tests /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /opt/naive \
             /opt/stash /opt/stash_oracle /app/solution /app/tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: reviewer-only or frozen path baked: $p"; rc=1; fi; \
    done; \
    hits="$(find / -xdev -type f \( -name 'holdout_corpus*' -o -name compute_reward.py \
              -o -name 'verifier-correctness-manifest.json' \) 2>/dev/null | head -20)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: baked frozen-surface files:"; echo "$hits"; rc=1; fi; \
    if grep -rslE 'strong_baseline_ce|ref_speedup|expected_base_ce' /app/submission /app/instruction.md 2>/dev/null | head -3 | grep -q .; then \
      echo "BUILD_ASSERT_FAIL: a calibrated anchor token is visible to the solver"; rc=1; fi; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS: no frozen surface, no reviewer recipe, no anchor visible"

# ---- runtime contract: entrypoint + elapsed/keepalive stub ----
RUN printf '#!/bin/bash\nset -e\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nSTART=$(date +%%s)\nwhile true; do echo "elapsed=$(( $(date +%%s) - START ))s"; sleep 60; done\n' > /app/timer.sh \
 && chmod +x /app/timer.sh

ENV BASE_MODEL_DIR=/app/base_model \
    A8_DATA_DIR=/data/datasets/a8_syscode \
    SUBMISSION_DIR=/app/submission \
    A8_PEFT_RO=/opt/peft_ro/src \
    E2E_INVENTORY_PATH=/opt/a8_inventory/baseline_inventory.tsv.gz \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /app/submission

# ---- build-time smoke (no GPU on the build host, so nothing CUDA-dependent is asserted) ----
RUN python - <<'PY'
import os, subprocess
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
assert torch.__version__.startswith("2.7.0"), torch.__version__
for f in ("/app/submission/train_adapter.py", "/app/submission/adapter_entry.py",
          "/app/base_model/model.safetensors", "/app/base_model/config.json",
          "/data/datasets/a8_syscode/train.txt", "/data/datasets/a8_syscode/val.txt",
          "/opt/peft_ro/src/peft/__init__.py", "/app/repo/src/peft/__init__.py"):
    assert os.path.isfile(f), f"missing: {f}"
tr = os.path.getsize("/data/datasets/a8_syscode/train.txt")
va = os.path.getsize("/data/datasets/a8_syscode/val.txt")
print(f"corpus: train {tr:,} B | val {va:,} B")
n = subprocess.run(["git", "-C", "/app/repo", "rev-list", "--count", "HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("OK: env, peft tree, frozen base, corpus and starter all in place")
PY

# ---- THE PRISTINE-IMAGE INVENTORY — must be the LAST content layer so it covers everything the
#      image ships. compute_reward.py FAILS CLOSED without it (an unenforceable byte cap must
#      never score); it finds this copy through E2E_INVENTORY_PATH, set above. ----
RUN mkdir -p /opt/a8_inventory \
 && find / -xdev -type f -printf '%i\t%s\t%p\n' 2>/dev/null | gzip -9 > /tmp/baseline_inventory.tsv.gz \
 && mv /tmp/baseline_inventory.tsv.gz /opt/a8_inventory/baseline_inventory.tsv.gz \
 && chown -R root:root /opt/a8_inventory \
 && chmod 0755 /opt/a8_inventory \
 && chmod 0444 /opt/a8_inventory/baseline_inventory.tsv.gz \
 && echo "[build] pristine inventory: $(zcat /opt/a8_inventory/baseline_inventory.tsv.gz | wc -l) files"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
