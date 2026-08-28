#!/usr/bin/env bash
# Run 5 reflow-idea FPO++ variants sequentially for ONE task.
# Training hyperparameters match scripts/run_main_benchmark.sh (FPO++ section).
#
# Usage:
#   GPU=0 TASK=Can CKPT_DIR=downloaded_checkpoints/95j3noe4_step_1000 \
#     bash scripts/run_task_reflow_baseline.sh
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
TASK="${TASK:?Set TASK (Can, Square, TwoArmBoxCleanup, TwoArmLiftTray, TwoArmThreading)}"
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to local BC checkpoint dir}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SEED="${SEED:-0}"
WANDB="${WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-0}"
START_VARIANT="${START_VARIANT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
TASK_SLUG="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"

if [[ ! -f "${CKPT_DIR}/policy/model.safetensors" ]]; then
  echo "ERROR: missing ${CKPT_DIR}/policy/model.safetensors"
  exit 1
fi

# Per-task FPO++ knobs (from run_main_benchmark.sh)
case "$TASK" in
  Can)
    TOTAL_TIMESTEPS=5000000
    DISCOUNT=0.99
    CLIP_COEF=0.02
    MAX_GRAD_NORM=5
    HUBER_DELTA=0.5
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  Square)
    TOTAL_TIMESTEPS=8000000
    DISCOUNT=0.995
    CLIP_COEF=0.01
    MAX_GRAD_NORM=25
    HUBER_DELTA=1
    CLAMP_LOGRATIO=None
    CLAMP_OLD_CFM=None
    ;;
  TwoArmBoxCleanup)
    TOTAL_TIMESTEPS=5000000
    DISCOUNT=0.995
    CLIP_COEF=0.03
    MAX_GRAD_NORM=5
    HUBER_DELTA=0.1
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  TwoArmLiftTray)
    TOTAL_TIMESTEPS=8000000
    DISCOUNT=0.999
    CLIP_COEF=0.03
    MAX_GRAD_NORM=1
    HUBER_DELTA=1
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  TwoArmThreading)
    TOTAL_TIMESTEPS=8000000
    DISCOUNT=0.999
    CLIP_COEF=0.01
    MAX_GRAD_NORM=1
    HUBER_DELTA=0.1
    CLAMP_LOGRATIO=5
    CLAMP_OLD_CFM=4
    ;;
  *)
    echo "Unknown TASK=$TASK"; exit 1
    ;;
esac

WANDB_ARGS=(--wandb_enable False)
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS=(--wandb_enable True --wandb_project flow-bc-fpo-finetuning)
fi

if [[ "$SAVE_EVAL_VIDEO" == "1" ]]; then
  SAVE_EVAL_VIDEO=True
else
  SAVE_EVAL_VIDEO=False
fi

COMMON=(
  torchrun --nproc_per_node=1 --master_port "$((29500 + GPU * 10))"
  finetune_online_rl.py
  --distributed True
  --base_policy_local_path "${CKPT_DIR}"
  --load_ema True
  "${WANDB_ARGS[@]}"
  --gradient_accumulation_steps 1
  --num_minibatches 8
  --log_freq 1
  --save_freq 2
  --rollout_freq 2
  --eval_num_episodes 200
  --data_collection_steps 1600
  --do_chunk_level_ppo True
  --eval_ema False
  --exploration_noise_std None
  --freeze_vision_encoder True
  --gae_lambda 0.99
  --n_action_samples 8
  --n_action_steps 16
  --num_envs 30
  --sampling_steps 10
  --spo_clip_coef 0.01
  --zero_sampling True
  --loss_mode fpo
  --task "${TASK}"
  --eval_env "${TASK}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --discount "${DISCOUNT}"
  --sde_sigma 0
  --cfm_loss_average_group_size 1
  --cfm_loss_use_huber True
  --cfm_loss_huber_delta "${HUBER_DELTA}"
  --clip_coef "${CLIP_COEF}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --clamp_logratio "${CLAMP_LOGRATIO}"
  --clamp_old_cfm_loss "${CLAMP_OLD_CFM}"
  --trust_region_mode ppo
  --reflow_ema_warmup_updates 500
  --seed "${SEED}"
  --save_eval_video "${SAVE_EVAL_VIDEO}"
)

VARIANTS=(reflow reward_aware adaptive_compute fpo_operator theory)
LOG_DIR="${ROOT}/logs/reflow_baseline_${TASK_SLUG}_gpu${GPU}_${STAMP}"
mkdir -p "$LOG_DIR"
echo "stamp=${STAMP}" > "${LOG_DIR}/stamp.txt"
echo "task=${TASK} gpu=${GPU} ckpt=${CKPT_DIR}" >> "${LOG_DIR}/stamp.txt"

started=false
if [[ -z "$START_VARIANT" ]]; then
  started=true
fi

for variant in "${VARIANTS[@]}"; do
  if [[ "$started" == "false" ]]; then
    if [[ "$variant" == "$START_VARIANT" ]]; then
      started=true
    else
      echo "[$(date '+%F %T')] SKIP ${variant} (before START_VARIANT=${START_VARIANT})"
      continue
    fi
  fi

  log_file="${LOG_DIR}/${variant}.log"
  out_dir="runs/flow_fpo_${TASK_SLUG}_${variant}_gpu${GPU}_${STAMP}"
  if [[ -n "$OUTPUT_DIR" && "$variant" == "${START_VARIANT}" ]]; then
    out_dir="$OUTPUT_DIR"
  fi

  latest_step=0
  resume_args=()
  if [[ -d "${out_dir}/checkpoints" ]]; then
    if [[ -d "${out_dir}/checkpoints/latest/policy" ]]; then
      resume_ckpt="${out_dir}/checkpoints/latest"
    else
      resume_ckpt="$(ls -d "${out_dir}/checkpoints"/step_* 2>/dev/null | sort -V | tail -1 || true)"
    fi
    if [[ -n "${RESUME_CKPT}" && "$variant" == "${START_VARIANT}" ]]; then
      resume_ckpt="${RESUME_CKPT}"
    fi
    if [[ -n "$resume_ckpt" && -d "${resume_ckpt}/policy" ]]; then
      resume_args=(--resume_checkpoint_path "${resume_ckpt}")
      if [[ "$resume_ckpt" == *"/latest" ]]; then
        latest_ckpt="$(ls -d "${out_dir}/checkpoints"/step_* 2>/dev/null | sort -V | tail -1 || true)"
        latest_step="${latest_ckpt##*/step_}"
        latest_step="${latest_step:-0}"
      else
        latest_step="${resume_ckpt##*/step_}"
      fi
      echo "[$(date '+%F %T')] Resume ${TASK}/${variant} from ${resume_ckpt} (step_${latest_step})"
    fi
  fi

  if [[ "$latest_step" -ge "$TOTAL_TIMESTEPS" && "$latest_step" -gt 0 ]]; then
    echo "[$(date '+%F %T')] SKIP ${variant}: already at step_${latest_step} (>= ${TOTAL_TIMESTEPS})"
    continue
  fi

  echo "[$(date '+%F %T')] GPU${GPU} ${TASK}/${variant} (baseline FPO++) -> ${log_file}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: CUDA_VISIBLE_DEVICES=${GPU} ${COMMON[*]} --fpo_variant ${variant} --output_dir ${out_dir} ${resume_args[*]}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    "${COMMON[@]}" \
      --fpo_variant "${variant}" \
      --experiment "finetune-fpo++-${TASK_SLUG}-${variant}" \
      --output_dir "${out_dir}" \
      "${resume_args[@]}" \
      >> "${log_file}" 2>&1
  echo "[$(date '+%F %T')] Finished ${TASK}/${variant}"
done

echo "[$(date '+%F %T')] All 5 variants done for ${TASK}. stamp=${STAMP}"
