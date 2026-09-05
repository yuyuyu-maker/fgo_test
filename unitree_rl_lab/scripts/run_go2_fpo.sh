#!/usr/bin/env bash
# Train FPO++ (baseline flow policy) on official Unitree-Go2-Velocity.
# FPO_VARIANT=baseline is FPO++. Other names (reflow, …) are research ideas, not FPO++.
# Usage:
#   GPU=1 FPO_VARIANT=baseline MAX_ITERS=1500 bash scripts/run_go2_fpo.sh
#   GPU=1 FPO_VARIANT=reflow NUM_ENVS=64 MAX_ITERS=3 bash scripts/run_go2_fpo.sh   # ideas smoke
#   RESUME=1 CHECKPOINT=/path/model_1499.pt FPO_VARIANT=kd_only bash scripts/run_go2_fpo.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="/workspace/fgo_test/isaaclab_experiments"
set +u
source "${ISAAC_ROOT}/source_env.sh"
set -eo pipefail

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1

GPU="${GPU:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1500}"
TASK="${TASK:-Unitree-Go2-Velocity}"
FPO_VARIANT="${FPO_VARIANT:-baseline}"
RUN_NAME="${RUN_NAME:-fpo_${FPO_VARIANT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
RESUME="${RESUME:-0}"
CHECKPOINT="${CHECKPOINT:-}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
TEACHER_KD_COEF="${TEACHER_KD_COEF:-}"
TEACHER_AUX_ZERO_X0_PROB="${TEACHER_AUX_ZERO_X0_PROB:-}"

WORKDIR="${WORKDIR:-/dev/shm/unitree_go2_fpo_${FPO_VARIANT}_gpu${GPU}}"
export TMPDIR="${TMPDIR:-/dev/shm/_unitree_go2_fpo_tmp_${FPO_VARIANT}_gpu${GPU}}"
mkdir -p "$TMPDIR"
WS_LOG="${ROOT}/logs/fpo"
mkdir -p "$WORKDIR" "$WS_LOG"
cd "$WORKDIR"

unset CUDA_VISIBLE_DEVICES || true
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

sync_logs() {
  if [[ -d "${WORKDIR}/logs/fpo" ]]; then
    mkdir -p "$WS_LOG"
    cp -a "${WORKDIR}/logs/fpo/." "$WS_LOG/" 2>/dev/null || true
  fi
}
# Sync less often to reduce bosfs I/O (override with SYNC_INTERVAL_SEC).
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-900}"
(
  while sleep "${SYNC_INTERVAL_SEC}"; do
    sync_logs
  done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_logs' EXIT

EXTRA_ARGS=()
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  EXTRA_ARGS+=(--experiment_name "${EXPERIMENT_NAME}")
fi
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--teacher_checkpoint "${TEACHER_CHECKPOINT}")
fi
if [[ -n "${TEACHER_KD_COEF}" ]]; then
  EXTRA_ARGS+=(--teacher_kd_coef "${TEACHER_KD_COEF}")
fi
if [[ -n "${TEACHER_AUX_ZERO_X0_PROB}" ]]; then
  EXTRA_ARGS+=(--teacher_aux_zero_x0_prob "${TEACHER_AUX_ZERO_X0_PROB}")
fi

if [[ "${FPO_VARIANT}" == "baseline" ]]; then
  echo "[$(date '+%F %T')] FPO++ Go2 (baseline) GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS}"
else
  echo "[$(date '+%F %T')] research idea variant=${FPO_VARIANT} (not FPO++) GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS}"
fi

python "${ROOT}/scripts/fpo/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --fpo_variant "$FPO_VARIANT" \
  --run_name "${RUN_NAME}" \
  --device "cuda:${GPU}" \
  "${EXTRA_ARGS[@]}"

sync_logs
echo "[$(date '+%F %T')] done. logs=${WS_LOG}"
ls -lt "$WS_LOG" | head
