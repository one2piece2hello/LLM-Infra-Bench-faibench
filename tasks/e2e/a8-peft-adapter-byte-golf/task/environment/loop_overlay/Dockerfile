# syntax=docker/dockerfile:1
# -loop16 overlay for e2e-a8-peft-adapter-byte-golf.
# FROM the open-source single-shot base image; add /opt/loop/ (submit + score_engine + sanitize +
# private eval-only dev proxy + baked pristine starter) and swap /app/instruction.md for the loop16
# statement. Pure additive layer — NO dependency change, NO change to /tests, compute_reward.py, or
# ANY graded code path. Preferably applied with `crane append` (~8s, zero GPU); this Dockerfile is the
# docker-build fallback. No GPU is needed to BUILD the overlay; the base needs 1 NVIDIA GPU to score.
#
# BUILD (context = THIS directory, environment/loop_overlay/):
#     cd <...>/a8-peft-adapter-byte-golf/task
#     docker build -f environment/Dockerfile             -t fai/e2e-a8-peft-adapter-byte-golf:oss .
#     docker build -f environment/loop_overlay/Dockerfile -t fai/e2e-a8-peft-adapter-byte-golf:oss-loop16 environment/loop_overlay
#
# 🔴 Structural red-line (modes MUST also be baked into the crane tar header, since `crane append`
# runs no RUN): submit.sh 0755 root; score_engine.sh + everything under /opt/loop/private/ +
# /opt/loop/baseline_submission/ is 0700/0600 root (unreadable by the solver's dropped uid). The
# private dir carries NO held-out corpus, NO reference recipe and NO calibrated anchor — the dev
# proxy measures a RAW gain on the PUBLIC dev corpus /data/datasets/a8_syscode/val.txt only.
ARG BASE_IMAGE=fai/e2e-a8-peft-adapter-byte-golf:oss
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

# 🔴 build-time assert: the overlay must carry NO calibrated anchor VALUE (strong_baseline_ce /
# ref_speedup / expected_base_ce) and NO reference recipe / held-out corpus.
RUN set -eu; rc=0; \
    if grep -rIEq "1\.454941697688918|1\.2326578617539896|1\.5657425161470404|1\.45494|1\.23265" /opt/loop; then \
      echo "BUILD_ASSERT_FAIL: calibrated anchor VALUE present in /opt/loop"; rc=1; fi; \
    hits="$(find /opt/loop -type f \( -name 'holdout_corpus*' -o -name 'verifier-correctness-manifest.json' \
              -o -path '*ceiling*' -o -path '*strong_baseline*' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: reference/anchor/held-out file baked into /opt/loop:"; echo "$hits"; rc=1; fi; \
    test -f /opt/loop/submit.sh && test -x /opt/loop/submit.sh || { echo "submit.sh missing/not exec"; rc=1; }; \
    test -f /opt/loop/private/dev_eval.py || { echo "dev-proxy dev_eval.py missing"; rc=1; }; \
    test "$rc" = 0 && echo "[build] loop16 overlay asserts PASS"
