#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
# Full preprocessing protocol used by the thesis.
SOURCES=(
  "data/backup/Small_All_Beauty_5.json"
  "data/backup/Small_Digital_Music_5.json"
  "data/Small_Toys_and_Games_5_dense10k.json"
)
PROCESSED_OUTPUTS=(
  "data/Small_All_Beauty_5_llama_filtered.json"
  "data/Small_Digital_Music_5_llama_filtered.json"
  "data/Small_Toys_and_Games_5_llama_filtered.json"
)
DOMAINS=(all_beauty digital_music toys_games)

RUN_DENSE_BUILD=true
DENSE_SOURCE_URL="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz"
DENSE_RAW_CACHE="data/raw/reviews_Toys_and_Games_5.json.gz"
DENSE_OUTPUT="data/Small_Toys_and_Games_5_dense10k.json"
DENSE_TARGET_SIZE=10000
DENSE_K_CORE=5
DENSE_SEED=2026
REBUILD_DENSE=false

RUN_LLM_PREPROCESSING=true
PREPROCESSING_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DEVICE="mps"                         # mps was used for the thesis; use auto/cuda/cpu elsewhere.
TEXT_FIELD="reviewText"
RATING_FIELD="overall"
ITEM_FIELD="asin"
BATCH_SIZE=16
REVIEW_BATCH_SIZE=128
MAX_LENGTH=512
MAX_SAMPLES=""                       # Empty means all rows.
REMOVE_MARGIN=0.5
EMPTY_REVIEW_POLICY="keep-original"
ADJUST_RATINGS=true
TRUST_REMOTE_CODE=false

# Existing processed JSON can be reused because split mapping does not require
# another LLM pass. Set false + OVERWRITE_PROCESSED=true for a full rebuild.
REUSE_EXISTING_PROCESSED=true
OVERWRITE_PROCESSED=false

RUN_SPLIT=true
SPLIT_SEED=42
TRAIN_RATIO=0.8
VALIDATION_RATIO=0.1
OVERWRITE_SPLITS=true
DRY_RUN=false
# ======================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ${#SOURCES[@]} -ne ${#PROCESSED_OUTPUTS[@]} || ${#SOURCES[@]} -ne ${#DOMAINS[@]} ]]; then
  echo "SOURCES, PROCESSED_OUTPUTS, and DOMAINS must have the same length" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_command() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  [[ "${DRY_RUN}" == true ]] || "$@"
}

if [[ "${RUN_DENSE_BUILD}" == true ]]; then
  COMMAND=(
    "${PYTHON}" preprocessing_reviews.py build-dataset
    --source-url "${DENSE_SOURCE_URL}"
    --raw-cache "${DENSE_RAW_CACHE}"
    --output "${DENSE_OUTPUT}"
    --target-size "${DENSE_TARGET_SIZE}"
    --k-core "${DENSE_K_CORE}"
    --seed "${DENSE_SEED}"
  )
  [[ "${REBUILD_DENSE}" == true ]] && COMMAND+=(--overwrite)
  run_command "${COMMAND[@]}"
fi

for index in "${!SOURCES[@]}"; do
  source_path="${SOURCES[index]}"
  processed_path="${PROCESSED_OUTPUTS[index]}"
  domain="${DOMAINS[index]}"

  if [[ "${RUN_LLM_PREPROCESSING}" == true ]]; then
    if [[ "${REUSE_EXISTING_PROCESSED}" == true && -s "${processed_path}" ]]; then
      echo "Reuse existing processed dataset: ${processed_path}"
    else
      COMMAND=(
        "${PYTHON}" preprocessing_reviews.py preprocess "${source_path}"
        --output "${processed_path}"
        --model "${PREPROCESSING_MODEL}"
        --domain "${domain}"
        --device "${DEVICE}"
        --text-field "${TEXT_FIELD}"
        --rating-field "${RATING_FIELD}"
        --item-field "${ITEM_FIELD}"
        --batch-size "${BATCH_SIZE}"
        --review-batch-size "${REVIEW_BATCH_SIZE}"
        --max-length "${MAX_LENGTH}"
        --remove-margin "${REMOVE_MARGIN}"
        --empty-review-policy "${EMPTY_REVIEW_POLICY}"
      )
      if [[ "${ADJUST_RATINGS}" == true ]]; then
        COMMAND+=(--adjust-ratings)
      else
        COMMAND+=(--no-adjust-ratings)
      fi
      [[ "${TRUST_REMOTE_CODE}" == true ]] && COMMAND+=(--trust-remote-code)
      [[ -n "${MAX_SAMPLES}" ]] && COMMAND+=(--max-samples "${MAX_SAMPLES}")
      [[ "${OVERWRITE_PROCESSED}" == true ]] && COMMAND+=(--overwrite)
      run_command "${COMMAND[@]}"
    fi
  fi

  if [[ "${RUN_SPLIT}" == true ]]; then
    COMMAND=(
      "${PYTHON}" preprocessing_reviews.py split "${processed_path}"
      --source "${source_path}"
      --seed "${SPLIT_SEED}"
      --train-ratio "${TRAIN_RATIO}"
      --validation-ratio "${VALIDATION_RATIO}"
    )
    [[ "${OVERWRITE_SPLITS}" == true ]] && COMMAND+=(--overwrite)
    run_command "${COMMAND[@]}"
  fi
done
