#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
DATASETS=(
  "data/Small_All_Beauty_5_llama_filtered.json"
  "data/Small_Digital_Music_5_llama_filtered.json"
  "data/Small_Toys_and_Games_5_llama_filtered.json"
)
# Full thesis matrix: 3 datasets x 4 models x 3 evaluation conditions.
# 1) raw reviewText / overall
# 2) filteredReviewText / overall
# 3) filteredReviewText / overall_new
MODELS=(qwen2.5_3b qwen-3b llama3.2_1b llama3.2_3b)
MODES=(pretrained pretrained-processed-mix pretrained-processed)
SEED=42
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
# Empty: each mode uses its own label; set overall/overall_new to override all.
GROUND_TRUTH_FIELD=""
MAX_LENGTH=512
MAX_NEW_TOKENS=8
INFERENCE_BATCH_SIZE=1
OUTPUT_DIR="experiments/outputs"
FORCE=true
DRY_RUN=false
KEEP_GOING=false
MAX_TRAIN_SAMPLES=""                 # Empty means no limit.
MAX_VAL_SAMPLES=""
MAX_TEST_SAMPLES=""
CONSTRAIN_RATING_OUTPUT=true
USE_4BIT=false
TRUST_REMOTE_CODE=false

# Arguments forwarded to every individual rating run. Examples:
# EXTRA_RATING_ARGS=(--train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1)
EXTRA_RATING_ARGS=()
# ======================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

# Backward-compatible optional preset override from the command line.
if [[ $# -gt 0 && "${1}" != --* ]]; then
  case "${1}" in
    main|ablation) MODES=(pretrained pretrained-processed-mix pretrained-processed) ;;
    baseline) MODES=(pretrained) ;;
    processed) MODES=(pretrained-processed) ;;
    *) echo "Unknown LLM preset: ${1}" >&2; exit 2 ;;
  esac
  shift
fi

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

COMMAND=(
  "${PYTHON}" -m experiments rating-matrix
  --datasets "${DATASETS[@]}"
  --models "${MODELS[@]}"
  --modes "${MODES[@]}"
  --seed "${SEED}"
  --max-length "${MAX_LENGTH}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --output-dir "${OUTPUT_DIR}"
)
[[ -n "${GROUND_TRUTH_FIELD}" ]] && COMMAND+=(--ground-truth-field "${GROUND_TRUTH_FIELD}")
[[ "${FORCE}" == true ]] && COMMAND+=(--force)
[[ "${DRY_RUN}" == true ]] && COMMAND+=(--dry-run)
[[ "${KEEP_GOING}" == true ]] && COMMAND+=(--keep-going)
COMMAND+=("$@")

RATING_ARGS=()
RATING_ARGS+=(
  --train-ratio "${TRAIN_RATIO}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
[[ -n "${MAX_TRAIN_SAMPLES}" ]] && RATING_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
[[ -n "${MAX_VAL_SAMPLES}" ]] && RATING_ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
[[ -n "${MAX_TEST_SAMPLES}" ]] && RATING_ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
if [[ "${CONSTRAIN_RATING_OUTPUT}" == true ]]; then
  RATING_ARGS+=(--constrain-rating-output)
else
  RATING_ARGS+=(--no-constrain-rating-output)
fi
[[ "${USE_4BIT}" == true ]] && RATING_ARGS+=(--use-4bit)
[[ "${TRUST_REMOTE_CODE}" == true ]] && RATING_ARGS+=(--trust-remote-code)
for extra_arg in ${EXTRA_RATING_ARGS[@]+"${EXTRA_RATING_ARGS[@]}"}; do
  RATING_ARGS+=("${extra_arg}")
done
[[ ${#RATING_ARGS[@]} -gt 0 ]] && COMMAND+=(-- "${RATING_ARGS[@]}")

exec "${COMMAND[@]}"
