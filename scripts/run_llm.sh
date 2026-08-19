#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

PRESET="${1:-main}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${PRESET}" in
  main)
    MODES=(pretrained pretrained-processed)
    ;;
  ablation)
    MODES=(pretrained pretrained-processed-mix pretrained-rating-only pretrained-processed)
    ;;
  baseline)
    MODES=(pretrained)
    ;;
  processed)
    MODES=(pretrained-processed)
    ;;
  *)
    echo "Unknown LLM preset: ${PRESET}" >&2
    echo "Choose: main, ablation, baseline, processed" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec "${PYTHON}" -m exp_llm rating-matrix --modes "${MODES[@]}" "$@"
