#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-test}"
MODEL_PATH="${MODEL_PATH:-output}"
TOKENIZER_PATH="${TOKENIZER_PATH:-output/DEP-tokenizer}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
GPU="${GPU:-0}"
EVAL_BACKEND="${EVAL_BACKEND:-auto}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_INPUT_LENGTH="${EVAL_MAX_INPUT_LENGTH:-1024}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_ABLATION_MODE="${EVAL_ABLATION_MODE:-none}"
DSP_ABLATION_MODE="${DSP_ABLATION_MODE:-full}"
ABLATION_LLM_PATH="${ABLATION_LLM_PATH:-}"
CAPTION_MODEL_NAME="${CAPTION_MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
CAPTION_CACHE_PATH="${CAPTION_CACHE_PATH:-}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-8}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-64}"

EFFECTIVE_DATASET_NAME="${RALAP_DATASET_NAME:-${DATASET_NAME:-}}"
if [[ -z "$EFFECTIVE_DATASET_NAME" && -n "${EVAL_DATA_DIR:-}" ]]; then
  case "${EVAL_DATA_DIR,,}" in
    *hatememes*) EFFECTIVE_DATASET_NAME="hatememes" ;;
    *food101*) EFFECTIVE_DATASET_NAME="food101" ;;
    *mmimdb*) EFFECTIVE_DATASET_NAME="mmimdb" ;;
  esac
fi

case "${EFFECTIVE_DATASET_NAME,,}" in
  hatememes) DEFAULT_PRIMARY_METRIC="auroc" ;;
  food101) DEFAULT_PRIMARY_METRIC="acc" ;;
  mmimdb) DEFAULT_PRIMARY_METRIC="multilabel" ;;
  *) DEFAULT_PRIMARY_METRIC="auto" ;;
esac
if [[ "$DEFAULT_PRIMARY_METRIC" == "auto" ]]; then
  export EVAL_PRIMARY_METRIC="${EVAL_PRIMARY_METRIC:-auto}"
else
  export EVAL_PRIMARY_METRIC="$DEFAULT_PRIMARY_METRIC"
fi

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.cache/huggingface/datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-$PWD/.cache/huggingface/evaluate}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$PWD/.cache/huggingface/metrics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export DISABLE_SAFETENSORS_CONVERSION="${DISABLE_SAFETENSORS_CONVERSION:-1}"
export RAG_EMBED_SIZE="${RAG_EMBED_SIZE:-768}"
export SPARSE_HIDDEN_SIZE="${SPARSE_HIDDEN_SIZE:-512}"
export PROMPT_TOKEN_COUNT="${PROMPT_TOKEN_COUNT:-3}"
export LABEL_ATTN_SIZE="${LABEL_ATTN_SIZE:-256}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

printf 'Evaluation dataset: %s\n' "${EFFECTIVE_DATASET_NAME:-auto-detect}"
printf 'Primary evaluation metric: %s\n' "${EVAL_PRIMARY_METRIC^^}"
printf 'DSP ablation mode: %s\n' "$DSP_ABLATION_MODE"

DATASET_ARGS=()
if [[ -n "${EVAL_DATA_DIR:-}" ]]; then
  DATASET_ARGS+=(--dataset_path "$EVAL_DATA_DIR")
elif [[ -n "${RALAP_DATASET_NAME:-}" ]]; then
  DATASET_ARGS+=(--dataset_name "$RALAP_DATASET_NAME")
elif [[ -n "${DATASET_NAME:-}" ]]; then
  DATASET_ARGS+=(--dataset_name "$DATASET_NAME")
fi

ABLATION_ARGS=(
  --ablation_mode "$EVAL_ABLATION_MODE"
  --dsp_ablation_mode "$DSP_ABLATION_MODE"
  --caption_model_name "$CAPTION_MODEL_NAME"
  --caption_batch_size "$CAPTION_BATCH_SIZE"
  --caption_max_new_tokens "$CAPTION_MAX_NEW_TOKENS"
)
if [[ -n "$ABLATION_LLM_PATH" ]]; then
  ABLATION_ARGS+=(--ablation_llm_path "$ABLATION_LLM_PATH")
fi
if [[ -n "$CAPTION_CACHE_PATH" ]]; then
  ABLATION_ARGS+=(--caption_cache_path "$CAPTION_CACHE_PATH")
fi

python -u model-eval.py \
  --mode infer \
  --split "$SPLIT" \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --backend "$EVAL_BACKEND" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --max_input_length "$EVAL_MAX_INPUT_LENGTH" \
  --limit "$EVAL_LIMIT" \
  --gpu "$GPU" \
  "${ABLATION_ARGS[@]}" \
  "${DATASET_ARGS[@]}"

python -u model-eval.py \
  --mode eval \
  --split "$SPLIT" \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --limit "$EVAL_LIMIT" \
  --gpu "$GPU" \
  "${ABLATION_ARGS[@]}" \
  "${DATASET_ARGS[@]}"
