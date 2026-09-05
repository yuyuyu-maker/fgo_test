#!/usr/bin/env bash
# Train Boston Dynamics Spot flat velocity with Gaussian PPO (RSL-RL baseline).
# Protocol aligned with isaaclab_experiments/scripts/train_spot.sh: 16384 envs, 1500 iters.
#
# Usage:
#   GPU=3 bash scripts/run_spot_ppo.sh
#   GPU=3 NUM_ENVS=64 MAX_ITERS=2 bash scripts/run_spot_ppo.sh   # smoke
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="/workspace/fgo_test/isaaclab_experiments"
ISAACLAB="${ISAAC_ROOT}/thirdparty/IsaacLab"
set +u
source "${ISAAC_ROOT}/source_env.sh"
set -eo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

GPU="${GPU:-3}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-1500}"
TASK="Isaac-Velocity-Flat-Spot-v0"
RUN_NAME="${RUN_NAME:-spot_ppo}"
WORKDIR="${WORKDIR:-/dev/shm/spot_ppo_gpu${GPU}}"
export TMPDIR="${TMPDIR:-/dev/shm/_spot_ppo_tmp_gpu${GPU}}"
WS_LOG="${ROOT}/logs/rsl_rl/spot_velocity"
mkdir -p "$TMPDIR" "$WORKDIR" "$WS_LOG"
cd "$WORKDIR"

unset CUDA_VISIBLE_DEVICES || true

sync_logs() {
  if [[ -d "${WORKDIR}/logs/rsl_rl/spot_velocity" ]]; then
    mkdir -p "$WS_LOG"
    cp -a "${WORKDIR}/logs/rsl_rl/spot_velocity/." "$WS_LOG/" 2>/dev/null || true
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

echo "[$(date '+%F %T')] Spot PPO GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS} task=${TASK}"

python "${ISAACLAB}/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --run_name "${RUN_NAME}" \
  --experiment_name spot_velocity \
  --device "cuda:${GPU}"

sync_logs
echo "[$(date '+%F %T')] done. logs=${WS_LOG}"
ls -lt "$WS_LOG" 2>/dev/null | head
