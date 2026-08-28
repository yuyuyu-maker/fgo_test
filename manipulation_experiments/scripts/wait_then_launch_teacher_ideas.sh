#!/usr/bin/env bash
# Wait until the current dual-GPU reflow baseline jobs finish, then train the
# remaining idea variants: reflow_teacher_kd + all_ideas_teacher_kd
# (old 5 variants are already in the running resume_reflow_baseline queue).
#
# Usage:
#   nohup bash scripts/wait_then_launch_teacher_ideas.sh \
#     > logs/wait_then_launch_teacher_ideas.nohup.log 2>&1 &
#   WAIT=0 bash scripts/wait_then_launch_teacher_ideas.sh   # skip wait (debug)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$TMPDIR" "$ROOT/logs"

STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
CKPT_ROOT="${ROOT}/downloaded_checkpoints"
IDEA_VARIANTS="${IDEA_VARIANTS:-reflow_teacher_kd all_ideas_teacher_kd}"
PIPE_LOG="${ROOT}/logs/wait_then_launch_teacher_ideas_${STAMP}.log"
WAIT="${WAIT:-1}"

resolve_ckpt_dir() {
  local ckpt_dir="$1"
  if [[ -f "${ckpt_dir}/policy/model.safetensors" ]]; then
    echo "${ckpt_dir}"
    return
  fi
  local nested
  nested="$(find "${ckpt_dir}" -type f -path '*/policy/model.safetensors' 2>/dev/null | head -1 || true)"
  if [[ -n "$nested" ]]; then
    dirname "$(dirname "$nested")"
    return
  fi
  echo "${ckpt_dir}"
}

current_jobs_busy() {
  pgrep -f 'scripts/resume_reflow_baseline.sh' >/dev/null 2>&1 && return 0
  pgrep -f 'finetune_online_rl.py' >/dev/null 2>&1 && return 0
  return 1
}

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "[$(date '+%F %T')] Wait-then-launch teacher ideas stamp=${STAMP} variants=${IDEA_VARIANTS}"

if [[ "$WAIT" == "1" ]]; then
  echo "[$(date '+%F %T')] Waiting for current reflow / finetune jobs to finish..."
  while current_jobs_busy; do
    echo "[$(date '+%F %T')] still busy (resume_reflow or finetune_online_rl); sleep 60s"
    sleep 60
  done
  echo "[$(date '+%F %T')] GPU jobs idle; launching teacher-idea variants"
else
  echo "[$(date '+%F %T')] WAIT=0; launching immediately"
fi

run_gpu0() {
  GPU=0 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$IDEA_VARIANTS" \
    TASK=Can CKPT_DIR="${CKPT_ROOT}/95j3noe4_step_1000" \
    TEACHER_CKPT="${TEACHER_CAN:-${CKPT_ROOT}/95j3noe4_step_1000}" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  GPU=0 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$IDEA_VARIANTS" \
    TASK=TwoArmBoxCleanup CKPT_DIR="${CKPT_ROOT}/lainyisy_step_10000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  echo "[$(date '+%F %T')] GPU0 teacher-idea pipeline done"
}

run_gpu1() {
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$IDEA_VARIANTS" \
    TASK=Square CKPT_DIR="${CKPT_ROOT}/trc7rbt0_step_110000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$IDEA_VARIANTS" \
    TASK=TwoArmLiftTray CKPT_DIR="${CKPT_ROOT}/ri0w9j39_step_20000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  THREAD_CKPT="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")"
  while [[ ! -f "${THREAD_CKPT}/policy/model.safetensors" ]]; do
    echo "[$(date '+%F %T')] waiting for Threading BC ckpt..."
    THREAD_CKPT="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")"
    sleep 30
  done
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$IDEA_VARIANTS" \
    TASK=TwoArmThreading CKPT_DIR="${THREAD_CKPT}" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  echo "[$(date '+%F %T')] GPU1 teacher-idea pipeline done"
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait "$PID0" "$PID1"
echo "[$(date '+%F %T')] ALL TEACHER-IDEA TASKS DONE stamp=${STAMP}" \
  | tee "${ROOT}/logs/wait_then_launch_teacher_ideas_${STAMP}.done"
