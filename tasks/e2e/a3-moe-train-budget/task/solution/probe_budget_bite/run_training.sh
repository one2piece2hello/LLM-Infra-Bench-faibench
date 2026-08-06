#!/bin/bash
# Entry contract (solver-owned; fully editable). The harness invokes this under a
# harness-owned wall-clock timer with these env set: SEED, PARAM_FLOOR, WALLCLOCK_SEC,
# OUT_CKPT, DATA_PATH, TOKENIZER_PATH, NANOGPT_REPO(=/app/repo).
#
# It MUST write a checkpoint to $OUT_CKPT within the budget (save periodically — the timer can
# stop you at any moment) and leave load_model_for_verification(checkpoint_path, device)
# importable from /app/submission/train_gpt.py. The model must clear the total-parameter FLOOR
# (the grader re-counts it).
#
# This starting version runs the provided dense baseline. Change anything you like — nanoGPT
# under /app/repo is fully editable too.
set -e
cd /app/submission
python3 train_gpt.py
