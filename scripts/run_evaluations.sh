#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 {prebert|llm|semantic} [mode] [options...]" >&2
  exit 2
fi
shift

case "${TARGET}" in
  prebert)
    exec "${SCRIPT_DIR}/run_prebert.sh" "$@"
    ;;
  llm)
    exec "${SCRIPT_DIR}/run_llm.sh" "$@"
    ;;
  semantic)
    exec "${SCRIPT_DIR}/run_semantic_retention.sh" "$@"
    ;;
  *)
    echo "Unknown target: ${TARGET}. Choose prebert, llm, or semantic." >&2
    exit 2
    ;;
esac
