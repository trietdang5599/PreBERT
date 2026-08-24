#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
# PreBERT-Rec feature-contribution ablation only.
DATASETS=(
  # Small_All_Beauty_5_llama_filtered
  # Small_Digital_Music_5_llama_filtered
  Small_Toys_and_Games_5_llama_filtered
)
SEEDS=(42)
SPLIT_PROFILE="8-1-1"  # 8-1-1 | 7-1-2
GROUND_TRUTH_FIELD="overall"  # overall | overall_new

# full: review + rating; without-review: rating only;
# without-rating: review only (no SVD/FM rating branch or rating biases).
REC_FEATURE_ABLATIONS=(without-review) #full without-review without-rating
STANDARDIZE_DEEP_FEATURES=true  # Fit StandardScaler on train entities only.

CLUSTER_METHODS=(birch)
BERT_MODELS=(modernbert-base)
FINE_TUNE_BERT=false
BALANCE_BERT_CLASSES=true
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.00001
REGRESSOR_ARCHITECTURE="linear"  # linear | fusion-mlp
MLP_HIDDEN_DIM=64
MLP_DROPOUT=0.1
K_TOPIC=40
NUM_WORDS=200
MAX_TOPICS_PER_WORD=2
CACHE_ROOT="chkpt/clustering_ablation"
OUTPUT_DIR="results/rec_feature_ablation"
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

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

COMMAND=(
  "${PYTHON}" -m experiments.evaluation
  --mode rec-feature-ablation
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --split-profile "${SPLIT_PROFILE}"
  --ground-truth-field "${GROUND_TRUTH_FIELD}"
  --rec-feature-ablations "${REC_FEATURE_ABLATIONS[@]}"
  --cluster-methods "${CLUSTER_METHODS[@]}"
  --bert-models "${BERT_MODELS[@]}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --regressor-architecture "${REGRESSOR_ARCHITECTURE}"
  --mlp-hidden-dim "${MLP_HIDDEN_DIM}"
  --mlp-dropout "${MLP_DROPOUT}"
  --num-topics "${K_TOPIC}"
  --num-words "${NUM_WORDS}"
  --max-topics-per-word "${MAX_TOPICS_PER_WORD}"
  --cache-root "${CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)
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
[[ "${FORCE_BERT}" == true ]] && COMMAND+=(--force-bert)
if [[ "${STANDARDIZE_DEEP_FEATURES}" == true ]]; then
  COMMAND+=(--standardize-deep-features)
else
  COMMAND+=(--no-standardize-deep-features)
fi
[[ "${FORCE_RESULTS}" == true ]] && COMMAND+=(--force-results)
[[ "${KEEP_GOING}" == true ]] && COMMAND+=(--keep-going)
[[ "${DRY_RUN}" == true ]] && COMMAND+=(--dry-run)

"${COMMAND[@]}" "$@"
