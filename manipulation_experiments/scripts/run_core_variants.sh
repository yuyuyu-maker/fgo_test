#!/usr/bin/env bash
# Dual-GPU core set: baseline FPO++ / reflow / all_ideas_teacher_kd.
# Teacher for all_ideas is the trained FPO++ baseline (eval-best), not BC.
# all_ideas writes *rlteacher* run dirs so the old BC-teacher runs are kept.
#
#   GPU0: Can -> TwoArmBoxCleanup
#   GPU1: Square -> TwoArmLiftTray -> TwoArmThreading
#
# Reuses STAMP=2026-08-26_22-18-01 so finished Can/Square reflow runs are skipped.
# Speed: 30 envs on Can/Square. Two-arm tasks use 12: pod memcg is 200GiB and
# 30+30 two-arm MuJoCo workers OOM (SIGKILL). Eval every 10 iters x 40 episodes.
# GPU1 spawn is delayed 90s so both jobs do not livelock bosfs at once.
#
# Usage:
#   nohup bash scripts/run_core_variants.sh > logs/run_core_variants.nohup.log 2>&1 &
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

STAMP="${STAMP:-2026-08-26_22-18-01}"
CKPT_ROOT="${ROOT}/downloaded_checkpoints"
CORE_VARIANTS="${CORE_VARIANTS:-baseline reflow all_ideas_teacher_kd}"
PIPE_LOG="${ROOT}/logs/run_core_variants_${STAMP}.log"

resolve_ckpt_dir() {
  local ckpt_dir="$1"
  if [[ -f "${ckpt_dir}/policy/model.safetensors" ]]; then
    echo "${ckpt_dir}"
    return
  fi
  local nested
  nested="$(find "${ckpt_dir}" -type f -path "*/policy/model.safetensors" 2>/dev/null | head -1 || true)"
  if [[ -n "$nested" ]]; then
    dirname "$(dirname "$nested")"
    return
  fi
  echo "${ckpt_dir}"
}

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "[$(date "+%F %T")] Core variants stamp=${STAMP} variants=${CORE_VARIANTS}"
echo "[$(date "+%F %T")] NUM_ENVS=${NUM_ENVS:-30} EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES:-40} ROLLOUT_FREQ=${ROLLOUT_FREQ:-10}"

run_gpu0() {
  GPU=0 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$CORE_VARIANTS" \
    TASK=Can CKPT_DIR="${CKPT_ROOT}/95j3noe4_step_1000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  GPU=0 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$CORE_VARIANTS" \
    TASK=TwoArmBoxCleanup CKPT_DIR="${CKPT_ROOT}/lainyisy_step_10000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  echo "[$(date "+%F %T")] GPU0 core pipeline done"
}

run_gpu1() {
  echo "[$(date "+%F %T")] GPU1 waiting 90s so env spawn does not contend with GPU0 on bosfs"
  sleep 90
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$CORE_VARIANTS" \
    TASK=Square CKPT_DIR="${CKPT_ROOT}/trc7rbt0_step_110000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$CORE_VARIANTS" \
    TASK=TwoArmLiftTray CKPT_DIR="${CKPT_ROOT}/ri0w9j39_step_20000" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  THREAD_CKPT="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")"
  while [[ ! -f "${THREAD_CKPT}/policy/model.safetensors" ]]; do
    echo "[$(date "+%F %T")] waiting for Threading BC ckpt..."
    THREAD_CKPT="$(resolve_ckpt_dir "${CKPT_ROOT}/6vqrn614_step_10000")"
    sleep 30
  done
  GPU=1 STAMP="$STAMP" SAVE_EVAL_VIDEO=0 \
    VARIANTS="$CORE_VARIANTS" \
    TASK=TwoArmThreading CKPT_DIR="${THREAD_CKPT}" \
    bash "${ROOT}/scripts/run_task_reflow_baseline.sh"
  echo "[$(date "+%F %T")] GPU1 core pipeline done"
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait "$PID0" "$PID1"
echo "[$(date "+%F %T")] ALL CORE VARIANTS DONE stamp=${STAMP}" \
  | tee "${ROOT}/logs/run_core_variants_${STAMP}.done"
