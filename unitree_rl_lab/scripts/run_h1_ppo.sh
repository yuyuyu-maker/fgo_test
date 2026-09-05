#!/usr/bin/env bash
# Train official Unitree-H1-Velocity Gaussian PPO.
#
# Usage:
#   GPU=1 bash scripts/run_h1_ppo.sh
#   GPU=1 MAX_ITERS=3 NUM_ENVS=64 bash scripts/run_h1_ppo.sh   # smoke
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="/workspace/fgo_test/isaaclab_experiments"
set +u
source "${ISAAC_ROOT}/source_env.sh"
set -eo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export UNITREE_MODEL_DIR="${UNITREE_MODEL_DIR:-/dev/shm/unitree_model}"

# Align budget with FPO++ H1 recipe (isaaclab_experiments/scripts/train_h1.sh):
#   NUM_ENVS=8192, MAX_ITERS=2000. Do not use 4096×3000 for main-table PPO.
GPU="${GPU:-1}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-2000}"
TASK="Unitree-H1-Velocity"
RUN_NAME="${RUN_NAME:-unitree_h1_ppo}"
USD="${UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd"

WORKDIR="${WORKDIR:-/dev/shm/unitree_h1_ppo_gpu${GPU}}"
export TMPDIR="${TMPDIR:-/dev/shm/_unitree_h1_ppo_tmp_gpu${GPU}}"
mkdir -p "$TMPDIR"
WS_LOG="${ROOT}/logs/rsl_rl/unitree_h1_velocity"
mkdir -p "$WORKDIR" "$WS_LOG"
cd "$WORKDIR"

unset CUDA_VISIBLE_DEVICES || true

sync_logs() {
  if [[ -d "${WORKDIR}/logs/rsl_rl/unitree_h1_velocity" ]]; then
    mkdir -p "$WS_LOG"
    cp -a "${WORKDIR}/logs/rsl_rl/unitree_h1_velocity/." "$WS_LOG/" 2>/dev/null || true
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

echo "[$(date '+%F %T')] Unitree H1 PPO GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS}"
echo "  usd=${USD}"

python "${ROOT}/scripts/rsl_rl/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --run_name "${RUN_NAME}" \
  --experiment_name unitree_h1_velocity \
  --device "cuda:${GPU}"

sync_logs
echo "[$(date '+%F %T')] done. logs=${WS_LOG}"
ls -lt "$WS_LOG" | head
