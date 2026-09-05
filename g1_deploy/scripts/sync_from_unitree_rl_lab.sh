#!/usr/bin/env bash
# Re-copy official deploy sources from sibling unitree_rl_lab (keeps local v1_ours).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${UNITREE_RL_LAB_DEPLOY:-${ROOT}/../unitree_rl_lab/deploy}"

if [[ ! -d "${SRC}/robots/g1_29dof" ]]; then
  echo "[sync] ERROR: not found: ${SRC}/robots/g1_29dof" >&2
  exit 1
fi

echo "[sync] From ${SRC}"
cp -a "${SRC}/include/." "${ROOT}/include/"
# Preserve local velocity/v1_ours
TMP_OURS="$(mktemp -d)"
if [[ -d "${ROOT}/robots/g1_29dof/config/policy/velocity/v1_ours" ]]; then
  cp -a "${ROOT}/robots/g1_29dof/config/policy/velocity/v1_ours" "${TMP_OURS}/"
fi
cp -a "${SRC}/robots/g1_29dof/." "${ROOT}/robots/g1_29dof/"
if [[ -d "${TMP_OURS}/v1_ours" ]]; then
  rm -rf "${ROOT}/robots/g1_29dof/config/policy/velocity/v1_ours"
  cp -a "${TMP_OURS}/v1_ours" "${ROOT}/robots/g1_29dof/config/policy/velocity/"
fi
rm -rf "${TMP_OURS}"

echo "[sync] Done. Rebuild: bash scripts/build_g1_ctrl.sh"
