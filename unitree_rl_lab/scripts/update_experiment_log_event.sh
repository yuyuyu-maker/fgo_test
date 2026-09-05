#!/usr/bin/env bash
# Append a short status line into EXPERIMENT_LOG.md when a tier finishes.
# Usage: bash scripts/update_experiment_log_event.sh "Spot kd_only done @ model_1499"
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/logs/fpo/EXPERIMENT_LOG.md"
MSG="${1:?message}"
{
  echo
  echo "- $(date '+%F %T') ${MSG}"
} >> "${LOG}"
echo "[update_experiment_log] ${MSG}"
