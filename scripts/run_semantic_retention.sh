#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
DATASETS=(
  "data/Small_All_Beauty_5_llama_filtered.json"
  "data/Small_Digital_Music_5_llama_filtered.json"
  "data/Small_Toys_and_Games_5_llama_filtered.json"
)
MODEL="sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_FIELD="filteredReviewText"
REFERENCE_FIELDS=(reviewText summary)
DEVICE="mps"                        # mps was used for the thesis; auto/cuda/cpu are also supported.
BATCH_SIZE=32
MAX_LENGTH=512
THRESHOLD=0.80
SEED=42
ONLY_CHANGED=true
MAX_SAMPLES=""                      # Empty means all rows.
OUTPUT_ROOT="experiments/semantic_outputs"
# ======================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

for dataset_path in "${DATASETS[@]}"; do
  dataset_name="$(basename "${dataset_path}" .json)"
  output_name="${dataset_name%_llama_filtered}"
  COMMAND=(
    "${PYTHON}" -m experiments semantic "${dataset_path}"
    --output-dir "${OUTPUT_ROOT}/${output_name}"
    --model "${MODEL}"
    --processed-field "${PROCESSED_FIELD}"
    --reference-fields "${REFERENCE_FIELDS[@]}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --max-length "${MAX_LENGTH}"
    --threshold "${THRESHOLD}"
    --seed "${SEED}"
  )
  if [[ "${ONLY_CHANGED}" == true ]]; then
    COMMAND+=(--only-changed)
  else
    COMMAND+=(--no-only-changed)
  fi
  [[ -n "${MAX_SAMPLES}" ]] && COMMAND+=(--max-samples "${MAX_SAMPLES}")
  "${COMMAND[@]}" "$@"
done
