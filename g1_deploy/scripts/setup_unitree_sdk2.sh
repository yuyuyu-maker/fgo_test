#!/usr/bin/env bash
# Clone + install official Unitree C++ SDK2 into /usr/local (system install).
# Repo: https://github.com/unitreerobotics/unitree_sdk2
#
# Needs: cmake, build-essential, and apt deps from README
#   sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/thirdparty/unitree_sdk2"
URL="${UNITREE_SDK2_GIT:-https://github.com/unitreerobotics/unitree_sdk2.git}"
PREFIX="${UNITREE_SDK2_PREFIX:-/usr/local}"

mkdir -p "${ROOT}/thirdparty"

if [[ -d "${DEST}/.git" ]]; then
  echo "[sdk2] Updating ${DEST}"
  git -C "${DEST}" pull --ff-only || true
else
  echo "[sdk2] Cloning ${URL} -> ${DEST}"
  git clone --depth 1 "${URL}" "${DEST}"
fi

mkdir -p "${DEST}/build"
cmake -S "${DEST}" -B "${DEST}/build" -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DBUILD_EXAMPLES=OFF
cmake --build "${DEST}/build" -j"$(nproc)"

if [[ -w "${PREFIX}" ]] || [[ -w "${PREFIX}/lib" ]]; then
  cmake --install "${DEST}/build"
else
  echo "[sdk2] Installing to ${PREFIX} (needs sudo)..."
  sudo cmake --install "${DEST}/build"
fi

echo "[sdk2] Done. Headers under ${PREFIX}/include, libs under ${PREFIX}/lib"
echo "[sdk2] Next: bash scripts/build_g1_ctrl.sh"
