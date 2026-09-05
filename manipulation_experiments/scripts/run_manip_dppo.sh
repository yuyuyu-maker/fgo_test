#!/usr/bin/env bash
# Single-GPU DPPO (Gaussian PPO-style) finetune on robomimic Can, from local BC ckpt.
# Matches run_main_benchmark.sh "DPPO Learned Noise" Can knobs; wandb off; no torchrun DDP.
#
# Usage:
#   GPU=3 bash scripts/run_manip_dppo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

GPU="${GPU:-3}"
SEED="${SEED:-0}"
TASK="${TASK:-Can}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/downloaded_checkpoints/95j3noe4_step_1000}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
NUM_ENVS="${NUM_ENVS:-30}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/dppo_learned_${TASK,,}_gpu${GPU}_${STAMP}}"
LOG_DIR="${ROOT}/logs/dppo_${TASK,,}_gpu${GPU}_${STAMP}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -f "${CKPT_DIR}/policy/model.safetensors" ]]; then
  echo "ERROR: missing ${CKPT_DIR}/policy/model.safetensors"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "[$(date '+%F %T')] DPPO Learned Can GPU=${GPU} ckpt=${CKPT_DIR} out=${OUTPUT_DIR}"

python finetune_online_rl.py \
  --distributed False \
  --device cuda:0 \
  --base_policy_local_path "${CKPT_DIR}" \
  --load_ema True \
  --wandb_enable False \
  --experiment "finetune-dppo-learned-can" \
  --output_dir "${OUTPUT_DIR}" \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 \
  --save_freq 10 \
  --rollout_freq 10 \
  --eval_num_episodes 40 \
  --data_collection_steps 1600 \
  --do_chunk_level_ppo True \
  --eval_ema False \
  --exploration_noise_std None \
  --freeze_vision_encoder True \
  --gae_lambda 0.99 \
  --n_action_samples 8 \
  --n_action_steps 16 \
  --num_envs "${NUM_ENVS}" \
  --sampling_steps 10 \
  --spo_clip_coef 0.01 \
  --zero_sampling True \
  --loss_mode dppo \
  --task "${TASK}" \
  --eval_env "${TASK}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --discount 0.99 \
  --sde_sigma 0.18 \
  --learn_sde_sigma True \
  --noise_injection_min 0.3 \
  --noise_injection_max 0.5 \
  --cfm_loss_average_group_size 1 \
  --cfm_loss_use_huber True \
  --cfm_loss_huber_delta 0.5 \
  --clip_coef 0.01 \
  --max_grad_norm 25 \
  --clamp_logratio 5 \
  --clamp_old_cfm_loss 4 \
  --trust_region_mode ppo \
  --save_eval_video False \
  --seed "${SEED}" \
  > "${LOG_DIR}/train.log" 2>&1

echo "[$(date '+%F %T')] DPPO done log=${LOG_DIR}/train.log"
