#!/usr/bin/env bash
# 前一步失败不再继续执行
set -euo pipefail

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" != "base" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return
  fi

  local dep_python="/home/dell/anaconda3/envs/dep/bin/python"
  if [[ -x "$dep_python" ]]; then
    printf '%s\n' "$dep_python"
    return
  fi

  printf '%s\n' "python"
}

PYTHON_BIN="$(resolve_python_bin)"

if [[ -n "${GPU:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

if [[ "${RUN_CREATE_PRINT_PYTHON:-0}" == "1" ]]; then
  printf '%s\n' "$PYTHON_BIN"
  exit 0
fi

printf 'Using Python interpreter: %s\n' "$PYTHON_BIN"
printf 'Using CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" create-dataset.py
