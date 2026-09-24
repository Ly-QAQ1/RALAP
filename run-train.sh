#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# Avoid DeepSpeed import-time CUDA_HOME probing failure on systems without nvcc/CUDA toolkit.
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-1}"
export GC_USE_REENTRANT="${GC_USE_REENTRANT:-0}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-output/DEP-tokenizer}"
export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-data/dataset_train}"
export RAG_EMBED_SIZE="${RAG_EMBED_SIZE:-768}"
export SPARSE_HIDDEN_SIZE="${SPARSE_HIDDEN_SIZE:-512}"
export PROMPT_TOKEN_COUNT="${PROMPT_TOKEN_COUNT:-3}"
export SAE_LOSS_WEIGHT="${SAE_LOSS_WEIGHT:-0.1}"
export SAE_SPARSE_WEIGHT="${SAE_SPARSE_WEIGHT:-0.001}"
export LABEL_LOSS_WEIGHT="${LABEL_LOSS_WEIGHT:-0.8}"
export LABEL_ATTN_SIZE="${LABEL_ATTN_SIZE:-256}"
export LABEL_ATTN_LOSS_WEIGHT="${LABEL_ATTN_LOSS_WEIGHT:-0.05}"
export OUTPUT_DIR="${OUTPUT_DIR:-output}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-}"
export REPORT_TO="${REPORT_TO:-wandb}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed/ds_z2_frozen_config.json}"

if ! "${PYTHON_BIN}" -c "import deepspeed" >/dev/null 2>&1; then
  echo "[ERROR] 当前 Python 环境中未安装 deepspeed。"
  echo "[ERROR] 请在当前环境安装后重试：pip install deepspeed"
  exit 1
fi

DEEPSPEED_ARGS=()
if [[ -n "${NUM_GPUS:-}" ]]; then
  DEEPSPEED_ARGS+=(--num_gpus "${NUM_GPUS}")
fi
if [[ -n "${DEEPSPEED_INCLUDE:-}" ]]; then
  DEEPSPEED_ARGS+=(--include "${DEEPSPEED_INCLUDE}")
fi
if [[ -n "${MASTER_PORT:-}" ]]; then
  DEEPSPEED_ARGS+=(--master_port "${MASTER_PORT}")
fi

FORCE_TORCHRUN=1 "${PYTHON_BIN}" -m deepspeed.launcher.runner "${DEEPSPEED_ARGS[@]}" model-train.py
