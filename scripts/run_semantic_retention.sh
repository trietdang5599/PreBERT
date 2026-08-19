#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

DATASETS=(
  Small_All_Beauty_5_llama_filtered
  Small_Digital_Music_5_llama_filtered
  Small_Toys_and_Games_5_llama_filtered
)

for dataset in "${DATASETS[@]}"; do
  output_name="${dataset%_llama_filtered}"
  "${PYTHON}" -m exp_llm semantic \
    "data/${dataset}.json" \
    --output-dir "exp_llm/semantic_outputs/${output_name}" \
    "$@"
done
