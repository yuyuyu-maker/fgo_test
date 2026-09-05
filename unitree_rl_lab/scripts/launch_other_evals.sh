#!/usr/bin/env bash
# Remaining evals + next tasks. Does not wait for DONE (avoids colliding with leftover waiters).
#   GPU0  G1-29dof official PPO (smoke then 4096/3000)
#   GPU2  manipulation Gaussian PPO (Can)
#   GPU3  Go2 quant/qual tracking
#   GPU1  when reflow_teacher_kd train exits: resume FPO++ + PPO teacher
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PIPE="${ROOT}/logs/fpo/go2_arch_ablation"
MANIP="/workspace/fgo_test/manipulation_experiments"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$PIPE"
exec > >(tee -a "${PIPE}/other_evals.out") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

FPO_PP_CKPT="${FPO_PP_CKPT:-${ROOT}/logs/fpo/unitree_go2_flat_flow/2026-09-02_20-16-21_fpo_baseline_unitree_mdp/model_1499.pt}"

log "other evals stamp=${STAMP}"

log "GPU0 G1 PPO smoke then full"
(
  set -eo pipefail
  GPU=0 NUM_ENVS=64 MAX_ITERS=2 RUN_NAME="g1_ppo_smoke_${STAMP}" \
    WORKDIR="/dev/shm/unitree_g1_ppo_smoke_gpu0" \
    bash "${ROOT}/scripts/run_g1_ppo.sh" \
    > "${PIPE}/g1_ppo_smoke.log" 2>&1
  log "G1 smoke ok; starting full PPO on GPU0"
  GPU=0 NUM_ENVS=4096 MAX_ITERS=3000 RUN_NAME="unitree_g1_ppo_${STAMP}" \
    WORKDIR="/dev/shm/unitree_g1_ppo_gpu0" \
    bash "${ROOT}/scripts/run_g1_ppo.sh" \
    > "${PIPE}/g1_ppo.log" 2>&1
) &
PID0=$!
sleep 45

log "GPU2 manipulation Gaussian PPO (Can)"
GPU=2 bash "${MANIP}/scripts/run_manip_gaussian_ppo.sh" \
  > "${PIPE}/manip_gaussian_ppo.log" 2>&1 &
PID2=$!
sleep 10

log "GPU3 Go2 quant/qual tracking"
GPU=3 bash "${ROOT}/scripts/eval_go2_quant_qual.sh" \
  > "${PIPE}/quant_qual_launch.log" 2>&1 &
PID3=$!

log "GPU1 waiter: resume FPO++ + PPO teacher after current cuda:1 train exits"
(
  while pgrep -af "scripts/fpo/train.py" | grep -q "cuda:1"; do
    sleep 30
  done
  sleep 20
  log "GPU1 free; resume FPO++ baseline + teacher"
  GPU=1 FPO_VARIANT=kd_only \
    EXPERIMENT_NAME=unitree_go2_baseline_plus_teacher \
    RUN_NAME="baseline_plus_teacher_${STAMP}" \
    RESUME=1 CHECKPOINT="${FPO_PP_CKPT}" \
    NUM_ENVS=4096 MAX_ITERS=1500 \
    WORKDIR="/dev/shm/unitree_go2_fpo_baseline_plus_teacher_gpu1" \
    bash "${ROOT}/scripts/run_go2_fpo.sh" \
    > "${PIPE}/train_baseline_plus_teacher.log" 2>&1
) &
PID1=$!

echo "${PID0} 0 g1_ppo" > "${PIPE}/other_evals_pids.txt"
echo "${PID1} 1 baseline_plus_teacher_waiter" >> "${PIPE}/other_evals_pids.txt"
echo "${PID2} 2 gaussian_ppo" >> "${PIPE}/other_evals_pids.txt"
echo "${PID3} 3 quant_qual" >> "${PIPE}/other_evals_pids.txt"
log "launched GPU0=${PID0} GPU1_waiter=${PID1} GPU2=${PID2} GPU3=${PID3}"

wait "$PID0" && log "OK GPU0 g1_ppo" || log "FAIL GPU0 g1_ppo"
wait "$PID1" && log "OK GPU1 baseline_plus_teacher" || log "FAIL GPU1 baseline_plus_teacher"
wait "$PID2" && log "OK GPU2 gaussian_ppo" || log "FAIL GPU2 gaussian_ppo"
wait "$PID3" && log "OK GPU3 quant_qual" || log "FAIL GPU3 quant_qual"
log "other evals finished"
touch "${PIPE}/OTHER_EVALS_DONE"
