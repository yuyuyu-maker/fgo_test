#!/usr/bin/env bash
# Train official Unitree-G1-29dof-Velocity Gaussian PPO (same RSL-RL stack as Go2).
# Requires G1 29-DoF USD from scripts/download_unitree_g1_usd.sh.
#
# Usage:
#   GPU=1 bash scripts/run_g1_ppo.sh   # official defaults: 4096 envs, 50000 iters
#   GPU=1 MAX_ITERS=3 NUM_ENVS=64 bash scripts/run_g1_ppo.sh   # smoke
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

GPU="${GPU:-2}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-50000}"
TASK="Unitree-G1-29dof-Velocity"
RUN_NAME="${RUN_NAME:-unitree_g1_ppo}"
USD="${UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
if [[ ! -f "$USD" ]]; then
  echo "ERROR: missing G1 USD at ${USD}"
  echo "Run: bash ${ROOT}/scripts/download_unitree_g1_usd.sh"
  exit 1
fi

WORKDIR="${WORKDIR:-/dev/shm/unitree_g1_ppo_gpu${GPU}}"
export TMPDIR="${TMPDIR:-/dev/shm/_unitree_g1_ppo_tmp_gpu${GPU}}"
mkdir -p "$TMPDIR"
WS_LOG="${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity"
mkdir -p "$WORKDIR" "$WS_LOG"
cd "$WORKDIR"

# Pin to one physical GPU as cuda:0 so resume across cards does not keep old device ids.
export CUDA_VISIBLE_DEVICES="${GPU}"
TRAIN_DEVICE="cuda:0"
sync_logs() {
  if [[ -d "${WORKDIR}/logs/rsl_rl/unitree_g1_29dof_velocity" ]]; then
    mkdir -p "$WS_LOG"
    cp -a "${WORKDIR}/logs/rsl_rl/unitree_g1_29dof_velocity/." "$WS_LOG/" 2>/dev/null || true
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

EXTRA_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
  [[ -n "${CHECKPOINT:-}" ]] && EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

echo "[$(date '+%F %T')] Unitree G1-29dof PPO physical_GPU=${GPU} (CUDA_VISIBLE_DEVICES) envs=${NUM_ENVS} iters=${MAX_ITERS} resume=${RESUME:-0}"
echo "  usd=${USD}"
[[ -n "${CHECKPOINT:-}" ]] && echo "  checkpoint=${CHECKPOINT}"

python "${ROOT}/scripts/rsl_rl/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --run_name "${RUN_NAME}" \
  --experiment_name unitree_g1_29dof_velocity \
  --device "${TRAIN_DEVICE}" \
  "${EXTRA_ARGS[@]}"

sync_logs
echo "[$(date '+%F %T')] done. logs=${WS_LOG}"
ls -lt "$WS_LOG" | head
