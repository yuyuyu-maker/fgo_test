#!/usr/bin/env bash
# FPO++ / full method on official Unitree-G1-29dof-Velocity.
# Real-robot / Unitree budget: match G1PPORunnerCfg — 4096 envs × 50000 iters.
# All G1 comparison arms (PPO / FPO++ / kd / Ours) must use this same budget.
# Usage:
#   GPU=0 FPO_VARIANT=baseline bash scripts/run_g1_fpo.sh
#   GPU=0 FPO_VARIANT=all_ideas_teacher_kd TEACHER_CHECKPOINT=/path/model_*.pt bash scripts/run_g1_fpo.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-50000}"
FPO_VARIANT="${FPO_VARIANT:-baseline}"
RUN_NAME="${RUN_NAME:-g1_fpo_${FPO_VARIANT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TASK="Unitree-G1-29dof-Velocity"
WORKDIR="${WORKDIR:-/dev/shm/unitree_g1_fpo_${FPO_VARIANT}_gpu${GPU}}"

EXTRA=()
[[ -n "${EXPERIMENT_NAME}" ]] && EXTRA+=(EXPERIMENT_NAME="${EXPERIMENT_NAME}")
[[ -n "${TEACHER_CHECKPOINT:-}" ]] && EXTRA+=(TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(RESUME=1)
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(CHECKPOINT="${CHECKPOINT}")

env GPU="$GPU" TASK="$TASK" FPO_VARIANT="$FPO_VARIANT" NUM_ENVS="$NUM_ENVS" MAX_ITERS="$MAX_ITERS" \
  RUN_NAME="$RUN_NAME" WORKDIR="$WORKDIR" UNITREE_MODEL_DIR=/dev/shm/unitree_model "${EXTRA[@]}" \
  bash "${ROOT}/scripts/run_go2_fpo.sh"
