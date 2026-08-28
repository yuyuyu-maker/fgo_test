#!/usr/bin/env bash
# Launch 5 reflow-idea FPO++ variants for ONE task on ONE GPU.
#
# NOTE: For paper-matched hyperparameters use run_task_reflow_baseline.sh instead
# (30 envs, full timesteps, rollout eval every 2 iters). This script kept for
# quick probes only.
#
# Usage:
#   GPU=0 TASK=Can CKPT_DIR=.../checkpoints/step_25 \
#     bash scripts/launch_task_reflow_ideas.sh
#   GPU=1 TASK=Square CKPT_DIR=.../checkpoints/step_25 \
#     bash scripts/launch_task_reflow_ideas.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$TMPDIR"

GPU="${GPU:-0}"
TASK="${TASK:-Can}"
CKPT_DIR="${CKPT_DIR:-}"
NUM_ENVS="${NUM_ENVS:-4}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SEED="${SEED:-0}"
WANDB="${WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
TASK_SLUG="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
  echo "ERROR: CKPT_DIR must be a BC checkpoint dir (contains policy/)"
  exit 1
fi
if [[ -d "${CKPT_DIR}/policy" ]]; then
  BASE_PATH="$CKPT_DIR"
else
  BASE_PATH="$CKPT_DIR"
fi

# Per-task FPO++ knobs (aligned with scripts/run_main_benchmark.sh)
case "$TASK" in
  Can)
    DISCOUNT=0.99
    CLIP_COEF=0.02
    MAX_GRAD_NORM=5
    HUBER_DELTA=0.5
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  Square)
    DISCOUNT=0.995
    CLIP_COEF=0.01
    MAX_GRAD_NORM=25
    HUBER_DELTA=1
    CLAMP_LOGRATIO=None
    CLAMP_OLD_CFM=None
    ;;
  TwoArmBoxCleanup)
    DISCOUNT=0.995
    CLIP_COEF=0.03
    MAX_GRAD_NORM=5
    HUBER_DELTA=0.1
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  TwoArmLiftTray)
    DISCOUNT=0.999
    CLIP_COEF=0.03
    MAX_GRAD_NORM=1
    HUBER_DELTA=1
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  TwoArmThreading)
    DISCOUNT=0.999
    CLIP_COEF=0.01
    MAX_GRAD_NORM=1
    HUBER_DELTA=0.1
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  *)
    echo "WARN: TASK=$TASK unknown; using Can defaults"
    DISCOUNT=0.99
    CLIP_COEF=0.02
    MAX_GRAD_NORM=5
    HUBER_DELTA=0.5
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
esac

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
  --task "${TASK}"
  --eval_env "${TASK}"
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
  --discount "${DISCOUNT}"
  --zero_sampling True
  --sde_sigma 0
  --cfm_loss_average_group_size 1
  --cfm_loss_use_huber True
  --cfm_loss_huber_delta "${HUBER_DELTA}"
  --clamp_old_cfm_loss "${CLAMP_OLD_CFM}"
  --clamp_logratio "${CLAMP_LOGRATIO}"
  --clip_coef "${CLIP_COEF}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --trust_region_mode ppo
  --seed "${SEED}"
  --log_freq 1
  --save_freq 50
  --rollout_freq 999999
  "${WANDB_ARGS[@]}"
)

VARIANTS=(reflow reward_aware adaptive_compute fpo_operator theory)
LOG_DIR="${ROOT}/logs/${TASK_SLUG}_reflow_ideas_gpu${GPU}_${STAMP}"
mkdir -p "$LOG_DIR"
PID_FILE="${LOG_DIR}/pids.txt"
: > "$PID_FILE"
echo "stamp=${STAMP}" > "${LOG_DIR}/stamp.txt"
echo "task=${TASK} gpu=${GPU} ckpt=${BASE_PATH}" >> "${LOG_DIR}/stamp.txt"

for variant in "${VARIANTS[@]}"; do
  log_file="${LOG_DIR}/${variant}.log"
  echo "[$(date '+%F %T')] GPU${GPU} ${TASK}/${variant} -> ${log_file}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: CUDA_VISIBLE_DEVICES=${GPU} ${COMMON[*]} --fpo_variant ${variant}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    nohup "${COMMON[@]}" \
      --fpo_variant "${variant}" \
      --experiment "finetune-fpo-${TASK_SLUG}-${variant}" \
      --output_dir "runs/flow_fpo_${TASK_SLUG}_${variant}_gpu${GPU}_${STAMP}" \
      > "${log_file}" 2>&1 &
  echo "$! ${GPU} ${TASK} ${variant}" >> "$PID_FILE"
  echo "  PID=$!"
  sleep 3
done

echo "[$(date '+%F %T')] Launched ${#VARIANTS[@]} ${TASK} variants on GPU${GPU}. Logs: ${LOG_DIR}"
cat "$PID_FILE"
