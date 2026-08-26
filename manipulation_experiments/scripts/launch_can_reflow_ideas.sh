#!/usr/bin/env bash
# Launch 5 Can FPO++ reflow-idea variants on GPU0/1.
#
# Requires a BC base checkpoint (from pretrain_flow_bc.py), e.g.:
#   CKPT_DIR=runs/flow_bc_can_.../checkpoints/step_1000 \
#     bash scripts/launch_can_reflow_ideas.sh
#
# Variants: reflow, reward_aware, adaptive_compute, fpo_operator, theory
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$TMPDIR" logs/can_reflow_ideas

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
CKPT_DIR="${CKPT_DIR:-}"
NUM_ENVS="${NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SEED="${SEED:-0}"
WANDB="${WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$CKPT_DIR" ]]; then
  # Prefer latest Can BC checkpoint under runs/
  CKPT_DIR="$(find runs -type d \( -path '*/flow_bc_can*/checkpoints/step_*' -o -path '*/checkpoints/step_*' \) 2>/dev/null | sort | tail -1 || true)"
fi
if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
  echo "ERROR: set CKPT_DIR to a Can BC checkpoint directory (…/checkpoints/step_N)"
  echo "  Example: CKPT_DIR=runs/flow_bc_can_gpu0_.../checkpoints/step_1000 bash scripts/launch_can_reflow_ideas.sh"
  exit 1
fi
# finetune expects base_policy_local_path pointing at the step dir that contains policy/
if [[ -d "${CKPT_DIR}/policy" ]]; then
  BASE_PATH="$CKPT_DIR"
elif [[ -d "${CKPT_DIR}/../policy" ]]; then
  BASE_PATH="$(cd "${CKPT_DIR}/.." && pwd)"
else
  BASE_PATH="$CKPT_DIR"
fi

WANDB_ARGS=(--wandb_enable False)
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS=(--wandb_enable True --wandb_project flow-bc-reflow-ideas)
fi

COMMON=(
  python finetune_online_rl.py
  --distributed False
  --debug True
  --base_policy_local_path "${BASE_PATH}"
  --load_ema True
  --task Can
  --eval_env Can
  --loss_mode fpo
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --num_envs "${NUM_ENVS}"
  --data_collection_steps 96
  --num_minibatches 4
  --update_epochs 4
  --do_chunk_level_ppo True
  --freeze_vision_encoder True
  --n_action_samples 8
  --n_action_steps 16
  --sampling_steps 10
  --cfm_loss_average_group_size 1
  --cfm_loss_use_huber True
  --cfm_loss_huber_delta 0.5
  --clamp_old_cfm_loss 4
  --clamp_logratio 5
  --clip_coef 0.02
  --max_grad_norm 5
  --trust_region_mode ppo
  --seed "${SEED}"
  --log_freq 1
  --save_freq 50
  --rollout_freq 999999
  "${WANDB_ARGS[@]}"
)

# GPU layout: 0 -> reflow, reward_aware, theory | 1 -> adaptive_compute, fpo_operator
declare -a VARIANTS=(reflow reward_aware adaptive_compute fpo_operator theory)
declare -a GPUS=(0 0 1 1 0)

LOG_DIR="${ROOT}/logs/can_reflow_ideas/${STAMP}"
mkdir -p "$LOG_DIR"
PID_FILE="${LOG_DIR}/pids.txt"
: > "$PID_FILE"
echo "stamp=${STAMP}" > "${LOG_DIR}/stamp.txt"
echo "ckpt=${BASE_PATH}" >> "${LOG_DIR}/stamp.txt"

launch_one() {
  local gpu="$1"
  local variant="$2"
  local log_file="${LOG_DIR}/${variant}.log"
  echo "[$(date '+%F %T')] Launching ${variant} on GPU ${gpu} -> ${log_file}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} ${COMMON[*]} --fpo_variant ${variant} --experiment finetune-fpo-can-${variant}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
    nohup "${COMMON[@]}" \
      --fpo_variant "${variant}" \
      --experiment "finetune-fpo-can-${variant}" \
      --output_dir "runs/flow_fpo_can_${variant}_${STAMP}" \
      > "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid} ${gpu} ${variant}" >> "$PID_FILE"
  echo "  PID=${pid}"
}

for i in "${!VARIANTS[@]}"; do
  launch_one "${GPUS[$i]}" "${VARIANTS[$i]}"
  sleep 2
done

echo "[$(date '+%F %T')] Launched ${#VARIANTS[@]} variants. Logs: ${LOG_DIR}"
cat "$PID_FILE"
