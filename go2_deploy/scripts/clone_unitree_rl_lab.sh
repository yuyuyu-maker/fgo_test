#!/usr/bin/env bash
# Clone official Unitree RL Lab (C++/ONNX deploy reference for Go2).
# https://github.com/unitreerobotics/unitree_rl_lab
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/thirdparty/unitree_rl_lab"
URL="${UNITREE_RL_LAB_GIT:-https://github.com/unitreerobotics/unitree_rl_lab.git}"
mkdir -p "${ROOT}/thirdparty"
if [[ -d "${DEST}/.git" ]]; then
  echo "[clone] pull ${DEST}"
  git -C "${DEST}" pull --ff-only || true
else
  echo "[clone] ${URL} -> ${DEST}"
  git clone --depth 1 "${URL}" "${DEST}"
fi
echo "[clone] done. Go2 C++ entry typically under:"
echo "  ${DEST}/deploy/robots/go2"
echo "See docs/DEPLOY_CPP_ONNX.md"
