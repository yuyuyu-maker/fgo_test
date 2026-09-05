#!/usr/bin/env bash
# FPO++ / full method on Isaac-Velocity-Flat-Spot-v0.
# Budget = FPO++ recipe (isaaclab_experiments/scripts/train_spot.sh):
#   NUM_ENVS=16384, MAX_ITERS=1500, learner 32 samples / 32 epochs, value_loss_coef=0.5.
# All Spot comparison arms (PPO / FPO++ / kd / Ours) must use this same budget.
# Do not lower envs/iters for OOM workarounds without an explicit override.
# Usage:
#   GPU=3 FPO_VARIANT=baseline bash scripts/run_spot_fpo.sh
#   GPU=3 FPO_VARIANT=all_ideas_teacher_kd TEACHER_CHECKPOINT=/path/model_1499.pt bash scripts/run_spot_fpo.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-3}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-1500}"
FPO_VARIANT="${FPO_VARIANT:-baseline}"
RUN_NAME="${RUN_NAME:-spot_fpo_${FPO_VARIANT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TASK="Isaac-Velocity-Flat-Spot-v0"
WORKDIR="${WORKDIR:-/dev/shm/spot_fpo_${FPO_VARIANT}_gpu${GPU}}"

EXTRA=()
[[ -n "${EXPERIMENT_NAME}" ]] && EXTRA+=(EXPERIMENT_NAME="${EXPERIMENT_NAME}")
[[ -n "${TEACHER_CHECKPOINT:-}" ]] && EXTRA+=(TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(RESUME=1)
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(CHECKPOINT="${CHECKPOINT}")

env GPU="$GPU" TASK="$TASK" FPO_VARIANT="$FPO_VARIANT" NUM_ENVS="$NUM_ENVS" MAX_ITERS="$MAX_ITERS" \
  RUN_NAME="$RUN_NAME" WORKDIR="$WORKDIR" "${EXTRA[@]}" \
  bash "${ROOT}/scripts/run_go2_fpo.sh"
