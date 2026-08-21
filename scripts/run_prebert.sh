#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
# Full thesis suite. Keep only selected entries to run a subset.
# MODES=(main preprocessing-ablation clustering-ablation encoder-ablation)
MODES=(main)
DATASETS=(
  Small_All_Beauty_5_llama_filtered
  Small_Digital_Music_5_llama_filtered
  Small_Toys_and_Games_5_llama_filtered
)
SEEDS=(42)
FEATURE_MODES=(full review-only rating-only raw)                 # Used when MODE='full', 'review-only', 'rating-only', 'raw'.
CLUSTER_METHODS=(birch)              # Used when MODE=custom.
BERT_MODELS=(modernbert-base)        # ModernBERT is the project default; used when MODE=custom.
FINE_TUNE_BERT=true                  # false = frozen pretrained BERT embeddings + VADER coarse score.
BALANCE_BERT_CLASSES=true             # Inverse-frequency loss when fine-tuning.
BATCH_SIZE=32
EPOCHS=100
NUM_WORDS=100
MAX_TOPICS_PER_WORD=2
CACHE_ROOT="chkpt/clustering_ablation"
OUTPUT_DIR="results/clustering_ablation"
FORCE_BERT=false
FORCE_RESULTS=false
KEEP_GOING=false
DRY_RUN=false
# ======================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

# A positional mode still overrides the configured suite for compatibility.
if [[ $# -gt 0 && "${1}" != --* ]]; then
  MODES=("${1}")
  shift
fi

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

for mode in "${MODES[@]}"; do
  COMMAND=(
    "${PYTHON}" -m experiments.evaluation
    --mode "${mode}"
    --datasets "${DATASETS[@]}"
    --seeds "${SEEDS[@]}"
    --feature-modes "${FEATURE_MODES[@]}"
    --cluster-methods "${CLUSTER_METHODS[@]}"
    --bert-models "${BERT_MODELS[@]}"
    --batch-size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --num-words "${NUM_WORDS}"
    --max-topics-per-word "${MAX_TOPICS_PER_WORD}"
    --cache-root "${CACHE_ROOT}"
    --output-dir "${OUTPUT_DIR}"
  )
  [[ "${FORCE_BERT}" == true ]] && COMMAND+=(--force-bert)
  if [[ "${FINE_TUNE_BERT}" == true ]]; then
    COMMAND+=(--fine-tune-bert)
  else
    COMMAND+=(--no-fine-tune-bert)
  fi
  if [[ "${BALANCE_BERT_CLASSES}" == true ]]; then
    COMMAND+=(--balance-bert-classes)
  else
    COMMAND+=(--no-balance-bert-classes)
  fi
  [[ "${FORCE_RESULTS}" == true ]] && COMMAND+=(--force-results)
  [[ "${KEEP_GOING}" == true ]] && COMMAND+=(--keep-going)
  [[ "${DRY_RUN}" == true ]] && COMMAND+=(--dry-run)
  "${COMMAND[@]}" "$@"
done
