#!/usr/bin/env bash
# Fine-tune Can (PickPlaceCan) with FPO++ online RL from a BC base policy.
#
# Important: Can is NOT trained from scratch. Paper pipeline is:
#   1) offline BC pretrain (flow matching) -> base checkpoint
#   2) online RL fine-tune (this script, FPO++)
#
# Usage:
#   # download base ckpt once (Google Drive folder from README)
#   pip install gdown
#   gdown --folder https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t \
#     -O downloaded_checkpoints
#
#   bash scripts/train_can_fpo.sh
#   GPU=0 SEED=0 WANDB=0 bash scripts/train_can_fpo.sh
#   DRY_RUN=1 bash scripts/train_can_fpo.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED="${SEED:-0}"
WANDB="${WANDB:-0}"          # 0 = local only; 1 = enable W&B
DRY_RUN="${DRY_RUN:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"
NUM_ENVS="${NUM_ENVS:-30}"   # paper main-benchmark setting

CKPT_DIR="${CKPT_DIR:-${ROOT}/downloaded_checkpoints/95j3noe4_step_1000}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/can_fpo_finetune"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/can_fpo_seed${SEED}_${STAMP}.log"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "ERROR: base policy checkpoint not found: ${CKPT_DIR}"
  echo "Download with:"
  echo "  pip install gdown"
  echo "  gdown --folder https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t -O downloaded_checkpoints"
  exit 1
fi

WANDB_ARGS=(--wandb_enable False)
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS=(--wandb_enable True --wandb_project flow-bc-fpo-finetuning)
fi

CMD=(
  torchrun --nproc_per_node="${NUM_GPUS}" finetune_online_rl.py
  --distributed True
  --base_policy_local_path "${CKPT_DIR}"
  --load-ema True
  --checkpoint_step step_1000
  --experiment "finetune-fpo++-can"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --task Can
  --eval_env Can
  --discount 0.99
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
  --num_envs "${NUM_ENVS}"
  --sampling_steps 10
  --spo_clip_coef 0.01
  --zero_sampling True
  --sde_sigma 0
  --cfm_loss_average_group_size 1
  --cfm_loss_use_huber True
  --cfm_loss_huber_delta 0.5
  --clip_coef 0.02
  --max_grad_norm 5
  --clamp_logratio 5
  --clamp_old_cfm_loss 4
  --trust_region_mode ppo
  --seed "${SEED}"
  "${WANDB_ARGS[@]}"
)

echo "[$(date '+%F %T')] Can FPO++ finetune  GPU=${GPU}  seed=${SEED}  num_envs=${NUM_ENVS}"
echo "  base ckpt -> ${CKPT_DIR}"
echo "  log       -> ${LOG_FILE}"
echo "  cmd       -> ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
