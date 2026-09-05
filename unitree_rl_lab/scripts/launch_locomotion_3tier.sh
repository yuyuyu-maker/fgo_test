#!/usr/bin/env bash
# Locomotion 3-tier protocol (no large arch ablation):
#   per robot: PPO -> FPO++ baseline -> ours (all_ideas_teacher_kd, PPO teacher)
#
# GPU0  G1: FPO++ (resume) -> ours
# GPU1  H1: PPO (running) -> FPO++ -> ours
# GPU2  manipulation Gaussian PPO chain (unchanged)
# GPU3  Spot: PPO (running) -> FPO++ -> ours
#
# G1 PPO already finished (model_2999).
#
# Usage:
#   nohup bash scripts/launch_locomotion_3tier.sh > logs/fpo/gpu_queues/3tier.out 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MANIP="/workspace/fgo_test/manipulation_experiments"
PIPE="${ROOT}/logs/fpo/gpu_queues"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$PIPE"
exec > >(tee -a "${PIPE}/3tier_${STAMP}.out") 2>&1

# Always log to stderr so $(wait_ppo_teacher) only captures the path.
log() { echo "[$(date '+%F %T')] $*" >&2; }

gpu_busy() {
  local g="$1"
  pgrep -af "python" 2>/dev/null | grep -E "train\.py|finetune_online_rl\.py" | grep -E \
    "cuda:${g}([, ]|$)|device cuda:${g}|CUDA_VISIBLE_DEVICES=${g}([, ]|$)" >/dev/null
}

wait_gpu_free() {
  local g="$1" label="$2"
  log "GPU${g} wait: ${label}"
  while gpu_busy "$g"; do sleep 45; done
  sleep 25
  log "GPU${g} free after ${label}"
}

wait_ppo_teacher() {
  local tag="$1" final_iter="$2"
  shift 2
  local ckpt="" path
  log "waiting ${tag} PPO teacher model_${final_iter}.pt"
  while [[ -z "$ckpt" ]]; do
    for path in "$@"; do
      ckpt="$(ls -1t ${path} 2>/dev/null | head -1 || true)"
      # strip accidental log noise; keep only a real .pt path
      ckpt="$(printf '%s\n' "$ckpt" | grep -E '\.pt$' | head -1 || true)"
      [[ -n "$ckpt" && -f "$ckpt" ]] && break
      ckpt=""
    done
    [[ -z "$ckpt" ]] && sleep 60
  done
  log "${tag} PPO teacher: ${ckpt}"
  printf '%s\n' "$ckpt"
}

sync_rsl_logs() {
  local src="$1" dst="$2"
  [[ -d "$src" ]] && mkdir -p "$dst" && cp -a "${src}/." "$dst/" 2>/dev/null || true
}

log "locomotion 3-tier stamp=${STAMP}"

# --- GPU0: G1 FPO++ (resume) -> ours ---
(
  set -eo pipefail
  G1_FPO_CKPT="/dev/shm/unitree_g1_fpo_baseline_gpu3/logs/fpo/unitree_g1_29dof_fpo_baseline/2026-09-03_22-24-57_g1_fpo_baseline_2026-09-03_22-24-10/model_300.pt"
  if [[ -f "$G1_FPO_CKPT" ]]; then
    log "GPU0 G1 FPO++ resume from model_300"
    GPU=0 FPO_VARIANT=baseline EXPERIMENT_NAME=unitree_g1_29dof_fpo_baseline \
      RESUME=1 CHECKPOINT="$G1_FPO_CKPT" MAX_ITERS=1500 \
      RUN_NAME="g1_fpo_baseline_resume_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_baseline_gpu0 \
      bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu0_g1_fpo_baseline.log" 2>&1
  else
    log "GPU0 G1 FPO++ from scratch"
    GPU=0 FPO_VARIANT=baseline EXPERIMENT_NAME=unitree_g1_29dof_fpo_baseline \
      RUN_NAME="g1_fpo_baseline_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_baseline_gpu0 \
      bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu0_g1_fpo_baseline.log" 2>&1
  fi
  sync_rsl_logs /dev/shm/unitree_g1_ppo_gpu0/logs/rsl_rl/unitree_g1_29dof_velocity \
    "${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity"
  teacher="$(wait_ppo_teacher G1 2999 \
    '/dev/shm/unitree_g1_ppo_gpu0/logs/rsl_rl/unitree_g1_29dof_velocity/*/model_2999.pt' \
    "${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity/*/model_2999.pt")"
  log "GPU0 G1 ours (all_ideas_teacher_kd)"
  GPU=0 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=unitree_g1_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$teacher" RUN_NAME="g1_all_ideas_${STAMP}" \
    WORKDIR=/dev/shm/unitree_g1_fpo_ours_gpu0 \
    bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu0_g1_ours.log" 2>&1
  log "OK GPU0 G1 FPO++ + ours"
) &
echo "$! gpu0_g1" >> "${PIPE}/3tier_pids.txt"

sleep 30

# --- GPU1: H1 PPO -> FPO++ -> ours ---
(
  set -eo pipefail
  wait_gpu_free 1 "H1 PPO"
  sync_rsl_logs /dev/shm/unitree_h1_ppo_gpu1/logs/rsl_rl/unitree_h1_velocity \
    "${ROOT}/logs/rsl_rl/unitree_h1_velocity"
  teacher="$(wait_ppo_teacher H1 2999 \
    '/dev/shm/unitree_h1_ppo_gpu1/logs/rsl_rl/unitree_h1_velocity/*/model_2999.pt' \
    "${ROOT}/logs/rsl_rl/unitree_h1_velocity/*/model_2999.pt")"
  log "GPU1 H1 FPO++ baseline"
  GPU=1 FPO_VARIANT=baseline EXPERIMENT_NAME=unitree_h1_flat_flow \
    RUN_NAME="h1_fpo_baseline_${STAMP}" WORKDIR=/dev/shm/unitree_h1_fpo_baseline_gpu1 \
    bash "${ROOT}/scripts/run_h1_fpo.sh" > "${PIPE}/gpu1_h1_fpo_baseline.log" 2>&1
  log "GPU1 H1 ours"
  GPU=1 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=unitree_h1_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$teacher" RUN_NAME="h1_all_ideas_${STAMP}" \
    WORKDIR=/dev/shm/unitree_h1_fpo_ours_gpu1 \
    bash "${ROOT}/scripts/run_h1_fpo.sh" > "${PIPE}/gpu1_h1_ours.log" 2>&1
  log "OK GPU1 H1 3-tier"
) &
echo "$! gpu1_h1" >> "${PIPE}/3tier_pids.txt"

sleep 20

# --- GPU2: manipulation (unchanged) ---
(
  set -eo pipefail
  wait_gpu_free 2 "Can Gaussian PPO"
  GPU=2 TASK=Square SEED=0 NUM_ENVS=30 TOTAL_TIMESTEPS=8000000 \
    CKPT_DIR="${MANIP}/downloaded_checkpoints/trc7rbt0_step_110000" \
    bash "${MANIP}/scripts/run_manip_gaussian_ppo.sh" > "${PIPE}/gpu2_square.log" 2>&1
  GPU=2 TASK=TwoArmBoxCleanup SEED=0 NUM_ENVS=30 TOTAL_TIMESTEPS=5000000 \
    CKPT_DIR="${MANIP}/downloaded_checkpoints/lainyisy_step_10000" \
    bash "${MANIP}/scripts/run_manip_gaussian_ppo.sh" > "${PIPE}/gpu2_box.log" 2>&1
  log "OK GPU2 manip"
) &
echo "$! gpu2_manip" >> "${PIPE}/3tier_pids.txt"

sleep 20

# --- GPU3: Spot PPO -> FPO++ -> ours ---
(
  set -eo pipefail
  wait_gpu_free 3 "Spot PPO"
  sync_rsl_logs /dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_velocity \
    "${ROOT}/logs/rsl_rl/spot_velocity"
  sync_rsl_logs /dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_flat \
    "${ROOT}/logs/rsl_rl/spot_velocity"
  teacher="$(wait_ppo_teacher Spot 1499 \
    '/dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_velocity/*/model_1499.pt' \
    '/dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_flat/*/model_1499.pt' \
    "${ROOT}/logs/rsl_rl/spot_velocity/*/model_1499.pt")"
  log "GPU3 Spot FPO++ baseline"
  GPU=3 FPO_VARIANT=baseline EXPERIMENT_NAME=spot_flat_flow \
    RUN_NAME="spot_fpo_baseline_${STAMP}" WORKDIR=/dev/shm/spot_fpo_baseline_gpu3 \
    bash "${ROOT}/scripts/run_spot_fpo.sh" > "${PIPE}/gpu3_spot_fpo_baseline.log" 2>&1
  log "GPU3 Spot ours"
  GPU=3 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=spot_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$teacher" RUN_NAME="spot_all_ideas_${STAMP}" \
    WORKDIR=/dev/shm/spot_fpo_ours_gpu3 \
    bash "${ROOT}/scripts/run_spot_fpo.sh" > "${PIPE}/gpu3_spot_ours.log" 2>&1
  log "OK GPU3 Spot 3-tier"
) &
echo "$! gpu3_spot" >> "${PIPE}/3tier_pids.txt"

log "3-tier queues armed: $(cat "${PIPE}/3tier_pids.txt" 2>/dev/null | tr '\n' ' ')"
wait
log "all 3-tier queues done"
touch "${PIPE}/TIER3_DONE"
