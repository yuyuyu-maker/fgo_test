#!/usr/bin/env bash
# Build g1_ctrl from this package (layout mirrors unitree_rl_lab/deploy).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT="${ROOT}/robots/g1_29dof"
ORT_VER="${ONNXRUNTIME_VERSION:-1.22.0}"
ORT="${ROOT}/thirdparty/onnxruntime-linux-x64-${ORT_VER}"

if [[ ! -f "${ORT}/lib/libonnxruntime.so.${ORT_VER}" ]]; then
  echo "[build] ONNX Runtime missing; downloading..."
  bash "${ROOT}/scripts/download_onnxruntime.sh"
fi

if ! echo '#include <unitree/robot/channel/channel_factory.hpp>' | \
     c++ -x c++ -E -I/usr/local/include - >/dev/null 2>&1; then
  echo "[build] WARN: unitree_sdk2 headers not found in /usr/local/include"
  echo "        Run: bash scripts/setup_unitree_sdk2.sh"
fi

mkdir -p "${ROBOT}/build"
cmake -S "${ROBOT}" -B "${ROBOT}/build"
cmake --build "${ROBOT}/build" -j"$(nproc)"

echo
echo "[build] OK: ${ROBOT}/build/g1_ctrl"
echo "  Sim2Sim:  cd ${ROBOT}/build && ./g1_ctrl"
echo "  Sim2Real: ./g1_ctrl --network eth0"
echo "  Runtime:  export LD_LIBRARY_PATH=${ORT}/lib:\${LD_LIBRARY_PATH}"
