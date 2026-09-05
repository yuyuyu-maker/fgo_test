#!/usr/bin/env bash
# Per-GPU follow-up queues (keeps all 4 cards busy with planned work).
#
# G1: no architecture ablation — PPO teacher -> full all_ideas_teacher_kd only (GPU0).
# GPU0  G1 PPO (running) -> G1 all_ideas_teacher_kd (PPO teacher)
# GPU1  Go2 baseline_plus_teacher (running) -> step-sweep eval -> H1 PPO
# GPU2  Can Gaussian PPO (running) -> Square -> TwoArmBoxCleanup
# GPU3  Spot Gaussian PPO baseline (not G1 FPO/ablation)
#
# Usage:
#   nohup bash scripts/launch_gpu_followups.sh > logs/fpo/gpu_queues/launcher.out 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MANIP="/workspace/fgo_test/manipulation_experiments"
PIPE="${ROOT}/logs/fpo/gpu_queues"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$PIPE"
exec > >(tee -a "${PIPE}/launcher_${STAMP}.out") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

gpu_busy() {
  local g="$1"
  pgrep -af "python" 2>/dev/null | grep -E "train\.py|finetune_online_rl\.py" | grep -E \
    "cuda:${g}([, ]|$)|device cuda:${g}|CUDA_VISIBLE_DEVICES=${g}([, ]|$)" >/dev/null
}

wait_gpu_free() {
  local g="$1"
  local label="$2"
  log "GPU${g} waiting for ${label} to finish"
  while gpu_busy "$g"; do
    sleep 45
  done
  sleep 25
  log "GPU${g} free after ${label}"
}

sync_g1_ppo_logs() {
  local src="/dev/shm/unitree_g1_ppo_gpu0/logs/rsl_rl/unitree_g1_29dof_velocity"
  local dst="${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    cp -a "${src}/." "$dst/" 2>/dev/null || true
  fi
}

wait_g1_ppo_teacher() {
  local ckpt=""
  log "waiting for G1 PPO teacher model_2999.pt"
  while [[ -z "$ckpt" ]]; do
    sync_g1_ppo_logs
    ckpt="$(ls -1t /dev/shm/unitree_g1_ppo_gpu0/logs/rsl_rl/unitree_g1_29dof_velocity/*/model_2999.pt 2>/dev/null | head -1 || true)"
    if [[ -z "$ckpt" ]]; then
      ckpt="$(ls -1t "${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity"/*/model_2999.pt 2>/dev/null | head -1 || true)"
    fi
    if [[ -z "$ckpt" ]]; then
      sleep 60
    fi
  done
  log "G1 PPO teacher ready: ${ckpt}"
  echo "$ckpt"
}

run_g1_fpo() {
  local gpu="$1"
  local variant="$2"
  local exp_name="$3"
  local run_name="$4"
  local teacher="${5:-}"
  local resume="${6:-0}"
  local ckpt="${7:-}"
  local logf="${PIPE}/gpu${gpu}_${run_name}.log"
  local extra=()
  [[ -n "$teacher" ]] && extra+=(TEACHER_CHECKPOINT="$teacher")
  [[ "$resume" == "1" ]] && extra+=(RESUME=1)
  [[ -n "$ckpt" ]] && extra+=(CHECKPOINT="$ckpt")
  log "GPU${gpu} FPO ${variant} -> ${logf}"
  env GPU="$gpu" \
    TASK=Unitree-G1-29dof-Velocity \
    FPO_VARIANT="$variant" \
    NUM_ENVS=4096 \
    MAX_ITERS=1500 \
    EXPERIMENT_NAME="$exp_name" \
    RUN_NAME="${run_name}_${STAMP}" \
    WORKDIR="/dev/shm/unitree_g1_fpo_${variant}_gpu${gpu}" \
    UNITREE_MODEL_DIR=/dev/shm/unitree_model \
    "${extra[@]}" \
    bash "${ROOT}/scripts/run_go2_fpo.sh" >"$logf" 2>&1
}

# ---------------------------------------------------------------------------
log "GPU follow-up queues stamp=${STAMP}"

# GPU0: G1 PPO -> all_ideas_teacher_kd
(
  set -eo pipefail
  wait_gpu_free 0 "G1 PPO"
  sync_g1_ppo_logs
  teacher="$(wait_g1_ppo_teacher)"
  run_g1_fpo 0 all_ideas_teacher_kd unitree_g1_all_ideas_ppo_teacher g1_all_ideas_teacher_kd "$teacher"
  log "OK GPU0 g1_all_ideas_teacher_kd"
) &
echo "$! gpu0" >> "${PIPE}/queue_pids.txt"

sleep 20

# GPU1: baseline_plus_teacher -> eval -> H1 PPO
(
  set -eo pipefail
  wait_gpu_free 1 "Go2 baseline_plus_teacher"
  log "GPU1 baseline_plus_teacher step-sweep"
  VARIANTS="baseline_plus_teacher" GPU=1 SUMMARY="${PIPE}/step_sweep_baseline_plus_teacher.txt" \
    bash "${ROOT}/scripts/eval_go2_arch_ablation.sh" \
    > "${PIPE}/gpu1_eval_baseline_plus_teacher.log" 2>&1 || log "WARN GPU1 step-sweep failed"
  python3 "${ROOT}/scripts/merge_go2_compare_table.py" > "${PIPE}/merge_after_bpt.log" 2>&1 || true
  log "GPU1 download H1 USD if needed"
  bash "${ROOT}/scripts/download_unitree_h1_usd.sh" > "${PIPE}/gpu1_h1_usd_download.log" 2>&1 || log "WARN H1 USD download failed"
  log "GPU1 H1 PPO full train"
  GPU=1 NUM_ENVS=4096 MAX_ITERS=3000 \
    RUN_NAME="unitree_h1_ppo_${STAMP}" \
    WORKDIR="/dev/shm/unitree_h1_ppo_gpu1" \
    bash "${ROOT}/scripts/run_h1_ppo.sh" \
    > "${PIPE}/gpu1_h1_ppo.log" 2>&1
  log "OK GPU1 h1_ppo"
) &
echo "$! gpu1" >> "${PIPE}/queue_pids.txt"

sleep 20

# GPU2: Can -> Square -> BoxCleanup Gaussian PPO
(
  set -eo pipefail
  wait_gpu_free 2 "Can Gaussian PPO"
  log "GPU2 Square Gaussian PPO"
  GPU=2 TASK=Square SEED=0 NUM_ENVS=30 TOTAL_TIMESTEPS=8000000 \
    CKPT_DIR="${MANIP}/downloaded_checkpoints/trc7rbt0_step_110000" \
    bash "${MANIP}/scripts/run_manip_gaussian_ppo.sh" \
    > "${PIPE}/gpu2_square_gaussian_ppo.log" 2>&1
  log "GPU2 TwoArmBoxCleanup Gaussian PPO"
  GPU=2 TASK=TwoArmBoxCleanup SEED=0 NUM_ENVS=30 TOTAL_TIMESTEPS=5000000 \
    CKPT_DIR="${MANIP}/downloaded_checkpoints/lainyisy_step_10000" \
    bash "${MANIP}/scripts/run_manip_gaussian_ppo.sh" \
    > "${PIPE}/gpu2_box_gaussian_ppo.log" 2>&1
  log "OK GPU2 manip square+box"
) &
echo "$! gpu2" >> "${PIPE}/queue_pids.txt"

sleep 20

# GPU3: Spot Gaussian PPO baseline (no G1 FPO / ablation on this card)
(
  set -eo pipefail
  wait_gpu_free 3 "prior GPU3 job"
  log "GPU3 Spot PPO baseline"
  GPU=3 RUN_NAME="spot_ppo_${STAMP}" WORKDIR="/dev/shm/spot_ppo_gpu3" \
    bash "${ROOT}/scripts/run_spot_ppo.sh" > "${PIPE}/gpu3_spot_ppo.log" 2>&1
  log "OK GPU3 spot_ppo"
) &
echo "$! gpu3" >> "${PIPE}/queue_pids.txt"

log "follow-up queues armed. pids=$(cat "${PIPE}/queue_pids.txt" 2>/dev/null | tr '\n' ' ')"
wait
log "all GPU follow-up queues finished"
touch "${PIPE}/FOLLOWUPS_DONE"
IPE}/FOLLOWUPS_DONE"
ouch "${PIPE}/FOLLOWUPS_DONE"
IPE}/FOLLOWUPS_DONE"
