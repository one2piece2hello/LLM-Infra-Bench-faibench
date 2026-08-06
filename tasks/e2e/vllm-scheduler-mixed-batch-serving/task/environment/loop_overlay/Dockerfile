# syntax=docker/dockerfile:1
# =============================================================================
# -loop16 overlay for e2e-vllm-scheduler-mixed-batch-serving.
# FROM the open-source single-shot base image; add /opt/loop/ (submit + score_engine + sanitize +
# private dev proxy + baked pristine starter) and swap /app/instruction.md for the loop16 statement.
# Pure additive layer — NO dependency change, NO change to /tests, compute_reward.py, or ANY graded
# code path. The graded surface (tests/: compute_reward.py + launch_baseline.sh + prompts.jsonl +
# reward_manifest.json + the timer) is uploaded fresh at scoring and is byte-identical to the base.
# Preferably applied with `crane append` (~8s, zero GPU); this Dockerfile is the docker-build
# fallback. No GPU is needed to BUILD the overlay; the base needs 1 NVIDIA GPU to RUN/score.
#
# BUILD (context = THIS directory, environment/loop_overlay/):
#     cd <...>/vllm-scheduler-mixed-batch-serving/task
#     docker build -f environment/Dockerfile             -t fai/vllm-scheduler-mixed-batch-serving:oss .
#     docker build -f environment/loop_overlay/Dockerfile -t fai/vllm-scheduler-mixed-batch-serving:oss-loop16 environment/loop_overlay
#
# 🔴 Structural red-line (modes MUST also be baked into the crane tar header, since `crane append`
# runs no RUN): submit.sh 0755 root; score_engine.sh + everything under /opt/loop/private/ +
# /opt/loop/baseline_submission/ is 0700/0600 root (unreadable by a dropped solver uid). Note the
# base task.toml ALREADY lists /opt/loop in forbidden_edit_paths, so the graded frozen-surface gate
# treats the overlay as read-only.
#
# 🔴 LEAK-FREE: the private dev proxy launches the candidate's own vLLM server on the same PUBLIC
# prompts as the model-visible /app/run_dev_bench.py and reports ABSOLUTE throughput. It bakes NO
# strong baseline (launch_baseline.sh), NO hidden prompts (prompts.jsonl), NO reward_manifest.json
# and NO calibrated ref_speedup.
ARG BASE_IMAGE=fai/vllm-scheduler-mixed-batch-serving:oss
FROM ${BASE_IMAGE}
SHELL ["/bin/bash","-c"]

RUN test ! -e /opt/loop || { echo "BUILD_ASSERT_FAIL: base already has /opt/loop"; exit 1; }

COPY opt_loop/ /opt/loop/
COPY instruction_loop16.md /app/instruction.md

RUN chown -R root:root /opt/loop \
 && chmod 0755 /opt/loop \
 && chmod 0755 /opt/loop/submit.sh \
 && chmod 0700 /opt/loop/score_engine.sh \
 && chmod -R go-rwx /opt/loop/private /opt/loop/baseline_submission \
 && chmod 0700 /opt/loop/private /opt/loop/baseline_submission \
 && find /opt/loop/private /opt/loop/baseline_submission -type f -exec chmod 0600 {} \; \
 && find /opt/loop/private /opt/loop/baseline_submission -type d -exec chmod 0700 {} \; \
 && chmod 0644 /app/instruction.md

# /logs/loop is created world-writable at runtime by submit.sh; nothing to bake.

# 🔴 build-time assert: NO calibrated anchor VALUE, NO hidden prompts, NO strong-baseline launcher,
# NO reward_manifest baked into /opt/loop.
RUN set -eu; rc=0; \
    if grep -rIEq "1\.2857108264648467|1\.28571" /opt/loop; then \
      echo "BUILD_ASSERT_FAIL: calibrated ref_speedup VALUE present in /opt/loop"; rc=1; fi; \
    hits="$(find /opt/loop -type f \( -name 'launch_baseline*.sh' -o -name 'prompts.jsonl' \
              -o -name 'reward_manifest.json' -o -name 'compute_reward.py' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface file baked into /opt/loop:"; echo "$hits"; rc=1; fi; \
    test -f /opt/loop/submit.sh && test -x /opt/loop/submit.sh || { echo "submit.sh missing/not exec"; rc=1; }; \
    test -f /opt/loop/private/tests/dev_eval.py || { echo "dev-proxy dev_eval.py missing"; rc=1; }; \
    test "$rc" = 0 && echo "[build] loop16 overlay asserts PASS"
