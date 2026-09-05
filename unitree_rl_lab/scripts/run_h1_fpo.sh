#!/usr/bin/env bash
# FPO++ / full method on official Unitree-H1-Velocity.
# Budget = FPO++ recipe (isaaclab_experiments/scripts/train_h1.sh):
#   NUM_ENVS=8192, MAX_ITERS=2000, learner 32 samples / 32 epochs.
# All H1 comparison arms (PPO / FPO++ / kd / Ours) must use this same budget.
# Do not use 4096 envs for main-table runs.
# Usage:
#   GPU=1 FPO_VARIANT=baseline bash scripts/run_h1_fpo.sh
#   GPU=1 FPO_VARIANT=all_ideas_teacher_kd TEACHER_CHECKPOINT=/path/model_1999.pt bash scripts/run_h1_fpo.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-1}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-2000}"
FPO_VARIANT="${FPO_VARIANT:-baseline}"
RUN_NAME="${RUN_NAME:-h1_fpo_${FPO_VARIANT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TASK="Unitree-H1-Velocity"
WORKDIR="${WORKDIR:-/dev/shm/unitree_h1_fpo_${FPO_VARIANT}_gpu${GPU}}"

EXTRA=()
[[ -n "${EXPERIMENT_NAME}" ]] && EXTRA+=(EXPERIMENT_NAME="${EXPERIMENT_NAME}")
[[ -n "${TEACHER_CHECKPOINT:-}" ]] && EXTRA+=(TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(RESUME=1)
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(CHECKPOINT="${CHECKPOINT}")

env GPU="$GPU" TASK="$TASK" FPO_VARIANT="$FPO_VARIANT" NUM_ENVS="$NUM_ENVS" MAX_ITERS="$MAX_ITERS" \
  RUN_NAME="$RUN_NAME" WORKDIR="$WORKDIR" UNITREE_MODEL_DIR=/dev/shm/unitree_model "${EXTRA[@]}" \
  bash "${ROOT}/scripts/run_go2_fpo.sh"
