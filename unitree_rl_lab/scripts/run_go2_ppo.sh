#!/usr/bin/env bash
# Train Unitree official Go2 PPO (task Unitree-Go2-Velocity).
# Usage:
#   GPU=1 MAX_ITERS=3000 bash scripts/run_go2_ppo.sh
#   GPU=1 MAX_ITERS=3 NUM_ENVS=64 bash scripts/run_go2_ppo.sh   # smoke
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="/workspace/fgo_test/isaaclab_experiments"
source "${ISAAC_ROOT}/source_env.sh"

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/dev/shm/_unitree_go2_ppo_tmp}"
mkdir -p "$TMPDIR"

GPU="${GPU:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-3000}"
TASK="Unitree-Go2-Velocity"
RUN_NAME="${RUN_NAME:-unitree_go2_ppo}"

WORKDIR="/dev/shm/unitree_go2_ppo"
WS_LOG="${ROOT}/logs/rsl_rl/unitree_go2_velocity"
mkdir -p "$WORKDIR" "$WS_LOG"
cd "$WORKDIR"

# Isaac Lab 2.1 AppLauncher has no --disable_fabric. Leave CUDA_VISIBLE_DEVICES
# unset so Vulkan/CUDA see the same GPU list; pin the device via --device.
unset CUDA_VISIBLE_DEVICES || true
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

sync_logs() {
  if [[ -d "${WORKDIR}/logs/rsl_rl/unitree_go2_velocity" ]]; then
    mkdir -p "$WS_LOG"
    cp -a "${WORKDIR}/logs/rsl_rl/unitree_go2_velocity/." "$WS_LOG/" 2>/dev/null || true
  fi
}

SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-900}"
(
  while sleep "${SYNC_INTERVAL_SEC}"; do
    sync_logs
  done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_logs' EXIT

echo "[$(date '+%F %T')] Unitree Go2 PPO GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS}"

python "${ROOT}/scripts/rsl_rl/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --run_name "${RUN_NAME}" \
  --device "cuda:${GPU}"

sync_logs
echo "[$(date '+%F %T')] done. logs=${WS_LOG}"
ls -lt "$WS_LOG" | head
