#!/usr/bin/env bash
# Dual-GPU plan:
#   GPU0: Can BC (save_freq=25)  -> then 5 Can reflow-idea variants
#   GPU1: Square BC (save_freq=25) -> then 5 Square reflow-idea variants
#
# Usage:
#   nohup bash scripts/run_dual_gpu_reflow_pipeline.sh >/dev/null 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$TMPDIR" logs/dual_reflow_pipeline runs

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="$ROOT/logs/dual_reflow_pipeline/pipeline_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

SAVE_FREQ="${SAVE_FREQ:-25}"
NUM_ENVS_FINETUNE="${NUM_ENVS_FINETUNE:-4}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
WANDB="${WANDB:-0}"

CAN_OUT="runs/flow_bc_can_gpu0_${STAMP}"
SQ_OUT="runs/flow_bc_square_gpu1_${STAMP}"
mkdir -p "$CAN_OUT" "$SQ_OUT" logs

COMMON_BC=(
  python pretrain_flow_bc.py
  --policy flowmatching
  --network_architecture mlp
  --mlp_dims "[1024, 1024, 1024]"
  --vision_backbone vit
  --flow_network_output_param u
  --cfm_loss_mode u
  --cfm_loss_use_huber False
  --cfm_loss_huber_delta 0.5
  --grad_clip_norm 25
  --horizon 16
  --n_action_steps 8
  --sampling_steps 10
  --batch_size 512
  --learning_rate 1e-4
  --lr_backbone 1e-5
  --weight_decay 1e-6
  --ema_power 0.995
  --enable_geometric_augmentations True
  --seed 3
  --num_workers 0
  --wandb_enable False
  --eval_num_envs 2
  --eval_num_episodes 10
  --log_freq 5
  --save_freq "${SAVE_FREQ}"
  --rollout_freq 999999
  --debug True
)

echo "[$(date '+%F %T')] Starting Can BC on GPU0 -> ${CAN_OUT}"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  nohup "${COMMON_BC[@]}" \
    --dataset ankile/robomimic-mh-can-image \
    --image_observation_keys robot0_eye_in_hand_image \
    --eval_env Can \
    --steps 500000 \
    --experiment flow_bc_can \
    --output_dir "${CAN_OUT}" \
    > "logs/pretrain_can_bc_gpu0_${STAMP}.log" 2>&1 &
CAN_PID=$!
echo "  Can BC PID=${CAN_PID}"

echo "[$(date '+%F %T')] Starting Square BC on GPU1 -> ${SQ_OUT}"
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
  nohup "${COMMON_BC[@]}" \
    --dataset ankile/robomimic-mh-square-image \
    --image_observation_keys agentview_image \
    --eval_env Square \
    --steps 1000000 \
    --max_num_episodes 100 \
    --experiment flow_bc_square \
    --output_dir "${SQ_OUT}" \
    > "logs/pretrain_square_bc_gpu1_${STAMP}.log" 2>&1 &
SQ_PID=$!
echo "  Square BC PID=${SQ_PID}"

echo "${CAN_PID} 0 can_bc" > "logs/dual_reflow_pipeline/pids_${STAMP}.txt"
echo "${SQ_PID} 1 square_bc" >> "logs/dual_reflow_pipeline/pids_${STAMP}.txt"

find_first_ckpt() {
  local run_dir="$1"
  find "$run_dir/checkpoints" -type d -name 'step_*' 2>/dev/null \
    | while read -r d; do
        [[ -d "$d/policy" ]] && echo "$d"
      done \
    | sort -V \
    | head -1
}

launch_variants() {
  local gpu="$1"
  local task="$2"
  local ckpt="$3"
  echo "[$(date '+%F %T')] Launching ${task} variants on GPU${gpu} from ${ckpt}"
  GPU="${gpu}" TASK="${task}" CKPT_DIR="${ckpt}" \
    NUM_ENVS="${NUM_ENVS_FINETUNE}" TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    WANDB="${WANDB}" STAMP="${STAMP}" \
    bash "$ROOT/scripts/launch_task_reflow_ideas.sh"
}

CAN_DONE=0
SQ_DONE=0

echo "[$(date '+%F %T')] Waiting for first BC checkpoints (save_freq=${SAVE_FREQ})..."
while [[ "$CAN_DONE" -eq 0 || "$SQ_DONE" -eq 0 ]]; do
  if [[ "$CAN_DONE" -eq 0 ]]; then
    CKPT="$(find_first_ckpt "$CAN_OUT" || true)"
    if [[ -n "${CKPT}" ]]; then
      echo "[$(date '+%F %T')] Can ckpt ready: ${CKPT}"
      if kill -0 "$CAN_PID" 2>/dev/null; then
        echo "  Stopping Can BC PID=${CAN_PID}"
        kill "$CAN_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$CAN_PID" 2>/dev/null || true
      fi
      launch_variants 0 Can "$CKPT"
      CAN_DONE=1
    elif ! kill -0 "$CAN_PID" 2>/dev/null; then
      echo "[$(date '+%F %T')] ERROR: Can BC died before checkpoint"
      CAN_DONE=1
    fi
  fi

  if [[ "$SQ_DONE" -eq 0 ]]; then
    CKPT="$(find_first_ckpt "$SQ_OUT" || true)"
    if [[ -n "${CKPT}" ]]; then
      echo "[$(date '+%F %T')] Square ckpt ready: ${CKPT}"
      if kill -0 "$SQ_PID" 2>/dev/null; then
        echo "  Stopping Square BC PID=${SQ_PID}"
        kill "$SQ_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$SQ_PID" 2>/dev/null || true
      fi
      launch_variants 1 Square "$CKPT"
      SQ_DONE=1
    elif ! kill -0 "$SQ_PID" 2>/dev/null; then
      echo "[$(date '+%F %T')] ERROR: Square BC died before checkpoint"
      SQ_DONE=1
    fi
  fi

  sleep 30
done

echo "[$(date '+%F %T')] Dual-GPU reflow pipeline finished launching."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pgrep -af 'finetune_online_rl|pretrain_flow_bc' | grep -v grep || true
