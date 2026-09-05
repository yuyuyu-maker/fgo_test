#!/usr/bin/env bash
# Step-sweep FPO++ on official Unitree-Go2-Velocity.
# Usage:
#   GPU=0 bash scripts/eval_go2_fpo_steps.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /workspace/fgo_test/isaaclab_experiments/source_env.sh

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
unset CUDA_VISIBLE_DEVICES || true
export TORCHDYNAMO_DISABLE=1

GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-256}"
FPO_VARIANT="${FPO_VARIANT:-baseline}"
CKPT="${CHECKPOINT:-${ROOT}/logs/fpo/unitree_go2_flat_flow/2026-09-02_20-16-21_fpo_baseline_unitree_mdp/model_1499.pt}"
OUT_DIR="$(dirname "$CKPT")"
LOG="${OUT_DIR}/step_sweep.log"
SUMMARY="${OUT_DIR}/step_sweep_summary.txt"

cd "$ROOT"
python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
  --task Unitree-Go2-Velocity \
  --headless \
  --device "cuda:${GPU}" \
  --num_envs "${NUM_ENVS}" \
  --eval_episodes 10 \
  --sampling_steps 64 32 16 8 4 1 \
  --eval_modes zero random \
  --fpo_variant "${FPO_VARIANT}" \
  --model "${FPO_VARIANT}=${CKPT}" \
  | tee "$LOG"

grep -A 40 "SUMMARY TABLE" "$LOG" | tee "$SUMMARY"
echo "wrote ${SUMMARY}"
