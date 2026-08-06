#!/bin/bash
# Entry contract (solver-owned; editable). The harness invokes this under a
# harness-owned wall-clock timer with these env set: SEED, PARAM_FLOOR,
# WALLCLOCK_SEC, OUT_CKPT, DATA_PATH, TOKENIZER_PATH, NANOGPT_REPO(=/app/repo).
# It MUST write a checkpoint to $OUT_CKPT within the budget and leave a
# load_model_for_verification(checkpoint_path, device) importable from
# /app/submission/train_gpt.py. The model must clear the total-parameter FLOOR
# (the grader re-counts it). This starting version runs the strong single-card
# MoE recipe; change anything you like (nanoGPT under /app/repo is fully editable).
set -e
cd /app/submission
python3 train_gpt.py
