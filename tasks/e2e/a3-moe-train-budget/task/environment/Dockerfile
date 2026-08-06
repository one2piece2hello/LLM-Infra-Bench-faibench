# syntax=docker/dockerfile:1
# =============================================================================
# e2e-a3-moe-train-budget — OPEN-SOURCE, self-contained task image
#
#   Task: beat a dense nanoGPT baseline at a FIXED compute budget (e.g. with a sparse MoE FFN).
#         The agent edits /app/submission/{train_gpt.py,run_training.sh}; the OPEN verifier under /tests
#         trains it under a FIXED budget and scores held-out quality (bits/byte).
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = the task/ directory (this Dockerfile lives in task/environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network: PyPI + github.com + huggingface.co)
#     cd <...>/a3-moe-train-budget/task
#     docker build -f environment/Dockerfile -t fai/e2e-a3-moe-train-budget:oss .
#   No GPU is needed to build. Building the corpus (below) takes ~15-30 min and the
#   resulting image is ~10 GB (torch + CUDA base + the tokenised corpus).
#   Non-H20 GPUs: pass your compute capability so self-built kernels target it:
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0  ...   # A100
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=8.9  ...   # L40S / 4090
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=9.0  ...   # H100 / H20 (default)
#
# RUN  (needs 1 NVIDIA GPU, driver >= 535)
#     docker run --rm --gpus all --shm-size=8g -it fai/e2e-a3-moe-train-budget:oss
#
# SCORE (inside the container; tests/ is MOUNTED, never baked into the image)
#     docker run --rm --gpus all --shm-size=8g \
#       -v "$PWD/tests:/tests:ro" fai/e2e-a3-moe-train-budget:oss bash /tests/test.sh   # -> /logs/verifier/reward.json
#
# 🔴 RE-CALIBRATE THE FROZEN ANCHORS ON YOUR BUILD  (read this before trusting a reward)
#   reward = min(1.0, ln((baseline_bpb / your_val_bpb) / ref_speedup) / ln(ref_speedup))
#            and 0 unless (baseline_bpb / your_val_bpb) > ref_speedup  — you must BEAT the anchor
#   with the FROZEN constants in tests/verifier-correctness-manifest.json:
#       baseline_bpb = 1.9550684328299388 ,  ref_speedup = 1.533747535323529
#   Both were measured on H20 **on the corpus built by the recipe below**. They are
#   valid for YOUR build only if (a) the GPU matches and (b) the corpus reproduces.
#   The scored path deliberately refuses VERIFIER_MODE (anti-cheat), so re-calibrate by
#   scoring the SHIPPED reference recipes AS THE CANDIDATE:
#     # 1) the strong baseline -> new baseline_bpb
#     docker run --rm --gpus all --shm-size=8g \
#       -v "$PWD/solution/baseline_dense_floor:/app/submission:ro" \
#       -v "$PWD/tests:/tests:ro" fai/e2e-a3-moe-train-budget:oss bash /tests/test.sh
#     # 2) the oracle -> new oracle_bpb ; ref_speedup = baseline_bpb / oracle_bpb
#     docker run --rm --gpus all --shm-size=8g \
#       -v "$PWD/solution/oracle_grouped_moe:/app/submission:ro" \
#       -v "$PWD/tests:/tests:ro" fai/e2e-a3-moe-train-budget:oss bash /tests/test.sh
#   Read val_bpb from /logs/verifier/benchmark_results.json (median of >=3 runs) and write
#   both numbers into tests/verifier-correctness-manifest.json -> reward_model.
#   (ref_speedup <= 1 or missing => HARD FAIL 0 by design.)
#
# VERSION ANCHORS (reproduce the original runtime; see task.toml)
#     CUDA 12.6 · Python 3.10 · torch 2.7.0 (cu126) · sentencepiece 0.2.1 · numpy>=1.26,<3
#
# WORKING TREE: /app/repo = nanoGPT + a vendored torchao package (~650 files)
#   The original private base materialised this tree; the task package ships no vendored
#   copy, so it is reconstructed here with `git clone` (nanoGPT for the training system,
#   pytorch/ao for the low-precision ops). Both are pure Python at this layer, so the tree
#   is a REAL directory (not a symlink) and edits inside it take effect directly — the
#   symlink trick is only needed for packages with compiled extensions.
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"
# Pinned to a concrete commit (nanoGPT publishes no tags) so the graded training system is
# byte-reproducible; a floating `master` would drift and make scores incomparable over time.
ARG NANOGPT_REF="3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
ARG TORCHAO_REF="v0.9.0"
ARG WIKITEXT_REPO="Salesforce/wikitext"
ARG WIKITEXT_CONFIG="wikitext-103-raw-v1"

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

# ---- venv. Keep this exact path: tests/test.sh pins /opt/kernelbench-venv/bin ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions
RUN pip install --no-cache-dir -U pip setuptools wheel

# ---- DL stack + the exact support libraries the training path imports, from PUBLIC indexes ----
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.7.0" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__; print('torch', torch.__version__)"
RUN printf 'torch==2.7.0\n' > /tmp/constraints.txt \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt \
        "sentencepiece==0.2.1" "pyarrow==17.0.0" "numpy>=1.26,<3" "einops==0.8.0" \
        "tqdm==4.66.5" "psutil" "huggingface_hub>=0.24,<1" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- /app/repo: the nanoGPT training system + a vendored torchao package.
#      Pinned by ARG so a future user can move either ref without editing this file.
#      No `|| true` anywhere: a silently empty /app/repo would destroy the task. ----
RUN set -eu; \
    mkdir -p /app/repo; cd /app/repo; git init -q; \
    git remote add origin https://github.com/karpathy/nanoGPT.git; \
    git fetch -q --depth 1 origin "${NANOGPT_REF}"; \
    git checkout -q FETCH_HEAD; \
    rm -rf /app/repo/.git /app/repo/assets; \
    git clone --depth 1 --branch "${TORCHAO_REF}" https://github.com/pytorch/ao.git /tmp/ao; \
    mv /tmp/ao/torchao /app/repo/torchao; \
    rm -rf /tmp/ao; \
    for m in /app/repo/model.py /app/repo/train.py /app/repo/configurator.py /app/repo/bench.py; do \
      test -f "$m" || { echo "BUILD_ASSERT_FATAL: /app/repo marker missing: $m"; exit 1; }; \
    done; \
    test -f /app/repo/torchao/__init__.py || { echo "BUILD_ASSERT_FATAL: torchao package missing"; exit 1; }; \
    n=$(find /app/repo -type f -not -path '*/.git/*' | wc -l); \
    echo "[build] /app/repo non-.git file count = $n (original base shipped ~650)"; \
    test "$n" -ge 300 || { echo "BUILD_ASSERT_FATAL: /app/repo has only $n files - clone mutilated"; exit 1; }; \
    t=$(find /app/repo -maxdepth 2 -type f -not -path '*/torchao/*' | wc -l); \
    test "$t" -ge 15 || { echo "BUILD_ASSERT_FATAL: only $t nanoGPT top-level files (<15)"; exit 1; }; \
    python -c 'import sys;sys.path.insert(0,"/app/repo");import model;print("[build] nanoGPT import OK",model.__file__)'

# ---- THE CORPUS. The original private base shipped /data/datasets/wikitext_bpe8k/{train,val}.bin
#      and /data/tokenizers/wikitext_bpe8k.model as opaque blobs. Shipping them in a public task
#      package is not an option (hundreds of MB), so they are REBUILT here from the public
#      WikiText-103-raw dataset with a deterministic recipe: fixed source revision, fixed
#      SentencePiece BPE settings (vocab 8192 -> fits uint16), single thread, no input shuffle.
#      The frozen reward anchors were calibrated on this corpus; the fingerprints printed at the
#      end let you confirm your rebuild matches (if it does not, re-calibrate — see the header).
RUN python - <<'PY'
import hashlib, os, pathlib, sys
import numpy as np, pyarrow.parquet as pq, sentencepiece as spm
from huggingface_hub import snapshot_download

REPO = os.environ.get("WIKITEXT_REPO", "Salesforce/wikitext")
CFG  = os.environ.get("WIKITEXT_CONFIG", "wikitext-103-raw-v1")
DATA, TOK = "/data/datasets/wikitext_bpe8k", "/data/tokenizers"
os.makedirs(DATA, exist_ok=True); os.makedirs(TOK, exist_ok=True)
EOS = 2

local = snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=[f"{CFG}/*"])
root = pathlib.Path(local, CFG)

def split(name):
    files = sorted(root.glob(f"{name}-*.parquet")) or sorted(root.glob(f"*{name}*.parquet"))
    if not files:
        sys.exit(f"BUILD_ASSERT_FATAL: no parquet for split {name!r} under {root}")
    out = []
    for f in files:
        out.extend(pq.read_table(f).column("text").to_pylist())
    return out

train, val = split("train"), split("validation")
print(f"[corpus] raw lines: train={len(train)} val={len(val)}")

spm_in = "/tmp/spm_input.txt"
with open(spm_in, "w", encoding="utf-8") as fh:
    for line in train:
        line = line.strip()
        if line:
            fh.write(line + "\n")

spm.SentencePieceTrainer.train(
    input=spm_in, model_prefix=f"{TOK}/wikitext_bpe8k",
    vocab_size=8192, model_type="bpe", character_coverage=0.9995,
    input_sentence_size=2000000, shuffle_input_sentence=False,
    num_threads=1, seed_sentencepiece_size=1000000,
    unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
)
os.remove(spm_in)
sp = spm.SentencePieceProcessor(model_file=f"{TOK}/wikitext_bpe8k.model")
assert sp.get_piece_size() == 8192, sp.get_piece_size()

def encode(lines, out, chunk=20000, flush=8_000_000):
    total, buf = 0, []
    with open(out, "wb") as fh:
        for i in range(0, len(lines), chunk):
            batch = [t.strip() for t in lines[i:i + chunk] if t.strip()]
            if not batch:
                continue
            for ids in sp.encode(batch):
                buf.extend(ids); buf.append(EOS)
            if len(buf) >= flush:
                np.asarray(buf, dtype=np.uint16).tofile(fh); total += len(buf); buf = []
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(fh); total += len(buf)
    return total

n_tr = encode(train, f"{DATA}/train.bin")
n_va = encode(val,   f"{DATA}/val.bin")
print(f"[corpus] tokens: train={n_tr:,} val={n_va:,}")
if n_tr < 30_000_000:
    sys.exit(f"BUILD_ASSERT_FATAL: train corpus only {n_tr} tokens - the graded token budget needs more")
if n_va < 100_000:
    sys.exit(f"BUILD_ASSERT_FATAL: val corpus only {n_va} tokens")

def sha(p, cap=None):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read() if cap is None else fh.read(cap))
    return h.hexdigest()

print("[corpus] FINGERPRINTS (compare against a known-good build before trusting the anchors)")
print("  tokenizer.model      sha256 =", sha(f"{TOK}/wikitext_bpe8k.model"))
print("  train.bin first 1MiB sha256 =", sha(f"{DATA}/train.bin", 1 << 20))
print("  val.bin              sha256 =", sha(f"{DATA}/val.bin"))
print("  train tokens =", n_tr, "| val tokens =", n_va)
PY

# ---- solver-owned submission: the working-but-beatable STARTER (not the answer) ----
RUN rm -rf /app/submission && mkdir -p /app/submission /app/.cache/torch_extensions /logs/verifier /logs/artifacts /app/.timer
COPY environment/submission/ /app/submission/
COPY instruction.md /app/instruction.md
RUN chmod +x /app/submission/run_training.sh

# ---- git baseline for the editable tree: ONE commit as the diff base, no remote, no reflog ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && git config --global --add safe.directory /app/submission \
 && cd /app/repo \
 && printf '__pycache__/\n*.pyc\nmodel_ckpt.pt\n*.tmp\n' > /app/repo/.gitignore \
 && git init -q . \
 && git add -A && git commit -qm "nanoGPT baseline system" \
 && rm -rf /app/repo/.git/logs \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && test ! -e /app/repo/.git/logs \
 && echo "[build] /app/repo: fresh 1-commit git, no remote, no reflog"

# ---- OPEN grading suite is NOT baked into the image (by design).
#   tests/ (test.sh + compute_reward.py + the held-out corpus + the frozen manifest carrying
#   baseline_bpb / ref_speedup) is MOUNTED FRESH at /tests when scoring. It ships in this
#   package — open-source, but deliberately outside the image so the solver can neither read
#   the held-out text nor the calibrated anchors while solving.

# ---- ANTI-LEAK ASSERTS: the anchors and the held-out surface must NOT be visible to the
#      solver. Any regression that re-introduces them dies here, at build time. ----
RUN set -eu; rc=0; \
    for p in /opt/verifier /opt/strong_baseline /opt/naive /opt/negative /opt/ceiling \
             /opt/oracle /opt/solution /app/solution /app/tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: leak path present: $p"; rc=1; fi; \
    done; \
    if [ -d /tests ] && [ -n "$(find /tests -mindepth 1 -type f 2>/dev/null | head -1)" ]; then \
      echo "BUILD_ASSERT_FAIL: /tests contains baked files"; rc=1; \
    fi; \
    for f in verifier-correctness-manifest.json compute_reward.py; do \
      hits=$(find / -xdev -name "$f" -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -3); \
      if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface file baked: $hits"; rc=1; fi; \
    done; \
    if grep -rslE 'ref_speedup|baseline_bpb|strong_baseline_bpb' /app 2>/dev/null | head -3 | grep -q .; then \
      echo "BUILD_ASSERT_FAIL: a calibrated anchor token is visible under /app"; rc=1; \
    fi; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS: no frozen surface, no reference recipe, no anchor under /app"

# ---- runtime contract: entrypoint + elapsed-time daemon ----
RUN printf '#!/bin/bash\nset -e\nmkdir -p /app/.timer\nnohup /app/timer.sh >/app/.timer/timer.log 2>&1 &\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nmkdir -p /app/.timer\nSTART=$(date +%%s)\nwhile true; do echo $(( $(date +%%s) - START )) > /app/.timer/elapsed; sleep 5; done\n' > /app/timer.sh \
 && chmod 755 /app/timer.sh

ENV NANOGPT_REPO=/app/repo \
    PYTHONPATH=/app/repo \
    SUBMISSION_DIR=/app/submission \
    REPO_DIR=/app/repo \
    DATA_PATH=/data/datasets/wikitext_bpe8k \
    TOKENIZER_PATH=/data/tokenizers/wikitext_bpe8k.model \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app

# ---- build-time smoke.
#      HARD: torch pin, nanoGPT + torchao import, the corpus is present and readable as
#            uint16, the tokenizer loads with 8192 pieces, the starter entry exists,
#            /app/repo has exactly one commit.
#      SOFT: nothing GPU-dependent is asserted (the build host has no card).
RUN python - <<'PY'
import os, subprocess, sys
import numpy as np, torch, sentencepiece as spm
sys.path.insert(0, "/app/repo")
print("torch", torch.__version__)
assert torch.__version__.startswith("2.7.0"), torch.__version__
import model as nanogpt_model, torchao
print("nanoGPT", nanogpt_model.__file__, "| torchao", torchao.__file__)

d = np.memmap("/data/datasets/wikitext_bpe8k/train.bin", dtype=np.uint16, mode="r")
v = np.memmap("/data/datasets/wikitext_bpe8k/val.bin", dtype=np.uint16, mode="r")
print("corpus tokens: train", f"{d.size:,}", "| val", f"{v.size:,}", "| max id", int(d[:1_000_000].max()))
assert d.size > 30_000_000 and v.size > 100_000
sp = spm.SentencePieceProcessor(model_file="/data/tokenizers/wikitext_bpe8k.model")
assert sp.get_piece_size() == 8192

for f in ("/app/submission/train_gpt.py", "/app/submission/run_training.sh"):
    assert os.path.isfile(f), f"submission entry missing: {f}"
n = subprocess.run(["git", "-C", "/app/repo", "rev-list", "--count", "HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("OK: env, working tree, corpus and starter all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
