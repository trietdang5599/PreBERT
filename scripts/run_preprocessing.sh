#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
# Full preprocessing protocol used by the thesis.
SOURCES=(
  "data/backup/Small_All_Beauty_5.json"
  "data/Small_Digital_Music_5_profile10k.json"
  "data/Small_Toys_and_Games_5_profile10k.json"
)
PROCESSED_OUTPUTS=(
  "data/Small_All_Beauty_5_llama_filtered.json"
  "data/Small_Digital_Music_5_llama_filtered.json"
  "data/Small_Toys_and_Games_5_llama_filtered.json"
)
DOMAINS=(all_beauty digital_music toys_games)

# Profile-aware 10K subsets preserve the full domains' interaction density
# much more closely than the former 5-core sampling strategy.
RUN_PROFILE_BUILDS=true
PROFILE_SOURCE_URLS=(
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz"
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz"
)
PROFILE_RAW_CACHES=(
  "data/raw/reviews_Digital_Music_5.json.gz"
  "data/raw/reviews_Toys_and_Games_5.json.gz"
)
PROFILE_OUTPUTS=(
  "data/Small_Digital_Music_5_profile10k.json"
  "data/Small_Toys_and_Games_5_profile10k.json"
)
PROFILE_NAMES=(digital-music toys-games)
PROFILE_TARGET_SIZE=10000
PROFILE_SEED=2026
REBUILD_PROFILE_SUBSETS=false

RUN_LLM_PREPROCESSING=true
PREPROCESSING_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DEVICE="mps"                         # mps was used for the thesis; use auto/cuda/cpu elsewhere.
TEXT_FIELD="reviewText"
RATING_FIELD="overall"
ITEM_FIELD="asin"
SEGMENTATION_MODE="hybrid"           # hybrid | sentence
MINIMUM_CLAUSE_WORDS=2               # Finer clause-level filtering.
BATCH_SIZE=16
REVIEW_BATCH_SIZE=128
MAX_LENGTH=512
MAX_SAMPLES=""                       # Empty means all rows.
REMOVE_MARGIN=0.05                   # Balanced-strict; lower values remove more uncertain text.
EMPTY_REVIEW_POLICY="keep-original"
ADJUST_RATINGS=true
TRUST_REMOTE_CODE=false
AUDIT_SAMPLE_SIZE=50
AUDIT_SEED=42
AUDIT_OUTPUT_ROOT="experiments/preprocessing_audits"

# Hybrid segmentation changes the LLM inputs, so rebuild the processed JSON
# once. After a successful rebuild, these can be changed back to true/false
# when only regenerating the deterministic split mapping.
REUSE_EXISTING_PROCESSED=false
OVERWRITE_PROCESSED=true

RUN_SPLIT=true
SPLIT_SEEDS=(41 42)
SPLIT_PROFILE="7-1-2"               # 8-1-1 | 7-1-2
TEST_RATING_FIELD="overall_new"      # overall | overall_new
OVERWRITE_SPLITS=true
DRY_RUN=false
# ======================================================================

# Make a command-line dry run cover both subset construction and the later
# preprocessing/splitting commands.
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

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

if [[ "${RUN_PROFILE_BUILDS}" == true ]]; then
  if [[ ${#PROFILE_SOURCE_URLS[@]} -ne ${#PROFILE_RAW_CACHES[@]} || ${#PROFILE_SOURCE_URLS[@]} -ne ${#PROFILE_OUTPUTS[@]} || ${#PROFILE_SOURCE_URLS[@]} -ne ${#PROFILE_NAMES[@]} ]]; then
    echo "PROFILE_SOURCE_URLS, PROFILE_RAW_CACHES, PROFILE_OUTPUTS, and PROFILE_NAMES must have the same length" >&2
    exit 2
  fi
  for index in "${!PROFILE_NAMES[@]}"; do
    COMMAND=(
      "${PYTHON}" preprocessing_reviews.py build-dataset
      --source-url "${PROFILE_SOURCE_URLS[index]}"
      --raw-cache "${PROFILE_RAW_CACHES[index]}"
      --output "${PROFILE_OUTPUTS[index]}"
      --target-size "${PROFILE_TARGET_SIZE}"
      --sampling-profile "${PROFILE_NAMES[index]}"
      --seed "${PROFILE_SEED}"
    )
    [[ "${REBUILD_PROFILE_SUBSETS}" == true ]] && COMMAND+=(--overwrite)
    run_command "${COMMAND[@]}"
  done
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
        --segmentation-mode "${SEGMENTATION_MODE}"
        --minimum-clause-words "${MINIMUM_CLAUSE_WORDS}"
        --batch-size "${BATCH_SIZE}"
        --review-batch-size "${REVIEW_BATCH_SIZE}"
        --max-length "${MAX_LENGTH}"
        --remove-margin "${REMOVE_MARGIN}"
        --empty-review-policy "${EMPTY_REVIEW_POLICY}"
        --audit-sample-size "${AUDIT_SAMPLE_SIZE}"
        --audit-seed "${AUDIT_SEED}"
        --audit-output "${AUDIT_OUTPUT_ROOT}/$(basename "${processed_path}" .json).csv"
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
    for split_seed in "${SPLIT_SEEDS[@]}"; do
      COMMAND=(
        "${PYTHON}" preprocessing_reviews.py split "${processed_path}"
        --source "${source_path}"
        --seed "${split_seed}"
        --split-profile "${SPLIT_PROFILE}"
        --test-rating-field "${TEST_RATING_FIELD}"
      )
      [[ "${OVERWRITE_SPLITS}" == true ]] && COMMAND+=(--overwrite)
      run_command "${COMMAND[@]}"
    done
  fi
done
