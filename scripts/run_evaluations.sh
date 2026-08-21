#!/usr/bin/env bash
set -euo pipefail

# ========================= USER CONFIGURATION =========================
TARGETS=(preprocessing prebert llm semantic)

# Extra arguments passed to the selected launcher. Most settings should be
# edited directly in that launcher's USER CONFIGURATION block.
RUNNER_ARGS=()
# ======================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A positional target still overrides the configured suite for compatibility.
if [[ $# -gt 0 && "${1}" != --* ]]; then
  TARGETS=("${1}")
  shift
fi

for target in "${TARGETS[@]}"; do
  case "${target}" in
    preprocessing) runner="${SCRIPT_DIR}/run_preprocessing.sh" ;;
    prebert) runner="${SCRIPT_DIR}/run_prebert.sh" ;;
    llm) runner="${SCRIPT_DIR}/run_llm.sh" ;;
    semantic) runner="${SCRIPT_DIR}/run_semantic_retention.sh" ;;
    *) echo "Unknown target: ${target}. Choose preprocessing, prebert, llm, or semantic." >&2; exit 2 ;;
  esac
  "${runner}" ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"} "$@"
done
