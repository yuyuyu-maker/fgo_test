#!/usr/bin/env bash
# Go2 component ablation vs already-trained all_ideas_teacher_kd (PPO teacher).
# Full method is skipped. Theory metrics are logs, always on.
#
# 4 GPUs:
#   reflow | reflow_teacher_kd (FPO++ + PPO teacher) | reward_aware | kd_only
# Then step-sweep (incl. existing baseline + full).
#
# Usage:
#   nohup bash scripts/launch_go2_arch_ablation.sh > logs/fpo/go2_arch_ablation/pipeline.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
PIPE_DIR="${ROOT}/logs/fpo/go2_arch_ablation"
mkdir -p "$PIPE_DIR"
echo "$STAMP" > "${PIPE_DIR}/stamp.txt"
OUT="${PIPE_DIR}/pipeline.out"
exec > >(tee -a "$OUT") 2>&1

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1500}"
STAGGER_SEC="${STAGGER_SEC:-45}"

WAVE1=(reflow reflow_teacher_kd reward_aware kd_only)
WAVE2=()

log() { echo "[$(date '+%F %T')] $*"; }

launch_one() {
  local gpu="$1"
  local variant="$2"
  local log_file="${PIPE_DIR}/train_${variant}.log"
  local run_name="arch_${STAMP}_${variant}"
  log "TRAIN ${variant} GPU=${gpu} envs=${NUM_ENVS} iters=${MAX_ITERS} -> ${log_file}"
  GPU="$gpu" FPO_VARIANT="$variant" RUN_NAME="$run_name" \
    NUM_ENVS="$NUM_ENVS" MAX_ITERS="$MAX_ITERS" \
    bash "${ROOT}/scripts/run_go2_fpo.sh" > "$log_file" 2>&1 &
  local pid=$!
  echo "${pid} ${gpu} ${variant}" >> "${PIPE_DIR}/pids.txt"
  log "  PID=${pid}"
}

wait_wave() {
  local ok=0
  local fail=0
  for pid_gpu_var in "$@"; do
    local pid="${pid_gpu_var%% *}"
    local rest="${pid_gpu_var#* }"
    local gpu="${rest%% *}"
    local variant="${rest#* }"
    if wait "$pid"; then
      log "OK   ${variant} (GPU ${gpu}, pid ${pid})"
      ok=$((ok + 1))
    else
      local rc=$?
      log "FAIL ${variant} (GPU ${gpu}, pid ${pid}, rc=${rc})"
      tail -n 40 "${PIPE_DIR}/train_${variant}.log" || true
      fail=$((fail + 1))
    fi
  done
  log "wave done ok=${ok} fail=${fail}"
  return 0
}

: > "${PIPE_DIR}/pids.txt"
log "Go2 architecture ablation stamp=${STAMP} envs=${NUM_ENVS} iters=${MAX_ITERS}"
log "Wave 1: ${WAVE1[*]}"

wave1_jobs=()
gpu=0
for variant in "${WAVE1[@]}"; do
  launch_one "$gpu" "$variant"
  wave1_jobs+=("$(tail -n 1 "${PIPE_DIR}/pids.txt")")
  gpu=$((gpu + 1))
  if [[ "$gpu" -lt 4 ]]; then
    log "stagger ${STAGGER_SEC}s before next Kit launch"
    sleep "$STAGGER_SEC"
  fi
done

wait_wave "${wave1_jobs[@]}"

if [[ ${#WAVE2[@]} -gt 0 ]]; then
  log "Wave 2: ${WAVE2[*]}"
  wave2_jobs=()
  gpu=0
  for variant in "${WAVE2[@]}"; do
    launch_one "$gpu" "$variant"
    wave2_jobs+=("$(tail -n 1 "${PIPE_DIR}/pids.txt")")
    gpu=$((gpu + 1))
    if [[ "$gpu" -lt ${#WAVE2[@]} ]]; then
      log "stagger ${STAGGER_SEC}s before next Kit launch"
      sleep "$STAGGER_SEC"
    fi
  done
  wait_wave "${wave2_jobs[@]}"
fi

log "All training waves finished. Starting step-sweep eval."
GPU=0 bash "${ROOT}/scripts/eval_go2_arch_ablation.sh" || log "WARN: eval script failed"

log "PIPELINE DONE stamp=${STAMP}"
touch "${PIPE_DIR}/DONE"
echo "summary: ${PIPE_DIR}"
