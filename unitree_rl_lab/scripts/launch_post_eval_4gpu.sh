#!/usr/bin/env bash
# After Go2 arch eval DONE: 4-GPU follow-up.
#   GPU0  FPO++ baseline + PPO teacher (resume FPO++ weights, train with kd_only)
#   GPU1  Go2 quantitative + qualitative (TB + tracking plots)
#   GPU2  G1-29dof PPO (after USD download / smoke)
#   GPU3  manipulation DPPO Learned on Can
#
# Usage:
#   nohup bash scripts/launch_post_eval_4gpu.sh > logs/fpo/go2_arch_ablation/post_eval_4gpu.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE_DIR="${ROOT}/logs/fpo/go2_arch_ablation"
MANIP="/workspace/fgo_test/manipulation_experiments"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="${PIPE_DIR}/post_eval_4gpu.out"
mkdir -p "$PIPE_DIR"
exec > >(tee -a "$OUT") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

DONE_FLAG="${PIPE_DIR}/DONE"
log "waiting for Go2 eval DONE at ${DONE_FLAG}"
while [[ ! -f "$DONE_FLAG" ]]; do
  sleep 60
done
log "eval DONE seen. waiting for GPUs to drain (no leftover train/eval python)"
for _ in $(seq 1 30); do
  if pgrep -af "fpo/train.py|eval_sampling_steps.py|rsl_rl/train.py" | grep -v "launch_post_eval" >/dev/null; then
    sleep 30
  else
    break
  fi
done
sleep 15

FPO_PP_CKPT="${FPO_PP_CKPT:-${ROOT}/logs/fpo/unitree_go2_flat_flow/2026-09-02_20-16-21_fpo_baseline_unitree_mdp/model_1499.pt}"
if [[ ! -f "$FPO_PP_CKPT" ]]; then
  FPO_PP_CKPT="$(ls -1t "${ROOT}"/logs/fpo/unitree_go2_flat_flow/*/model_1499.pt 2>/dev/null | head -1 || true)"
fi

log "GPU0 FPO++ baseline + teacher  resume=${FPO_PP_CKPT}"
GPU=0 FPO_VARIANT=kd_only \
  EXPERIMENT_NAME=unitree_go2_baseline_plus_teacher \
  RUN_NAME="baseline_plus_teacher_${STAMP}" \
  RESUME=1 CHECKPOINT="${FPO_PP_CKPT}" \
  NUM_ENVS=4096 MAX_ITERS=1500 \
  WORKDIR="/dev/shm/unitree_go2_fpo_baseline_plus_teacher_gpu0" \
  bash "${ROOT}/scripts/run_go2_fpo.sh" \
  > "${PIPE_DIR}/train_baseline_plus_teacher.log" 2>&1 &
PID0=$!
sleep 45

log "GPU1 Go2 quant/qual"
GPU=1 bash "${ROOT}/scripts/eval_go2_quant_qual.sh" \
  > "${PIPE_DIR}/quant_qual_launch.log" 2>&1 &
PID1=$!
sleep 20

log "GPU2 G1 PPO (USD download if needed, then smoke, then train)"
(
  set -eo pipefail
  bash "${ROOT}/scripts/download_unitree_g1_usd.sh"
  GPU=2 NUM_ENVS=64 MAX_ITERS=2 RUN_NAME="g1_ppo_smoke_${STAMP}" \
    WORKDIR="/dev/shm/unitree_g1_ppo_smoke_gpu2" \
    bash "${ROOT}/scripts/run_g1_ppo.sh" \
    > "${PIPE_DIR}/g1_ppo_smoke.log" 2>&1
  log "G1 smoke ok; starting full PPO"
  GPU=2 NUM_ENVS=4096 MAX_ITERS=3000 RUN_NAME="unitree_g1_ppo_${STAMP}" \
    WORKDIR="/dev/shm/unitree_g1_ppo_gpu2" \
    bash "${ROOT}/scripts/run_g1_ppo.sh" \
    > "${PIPE_DIR}/g1_ppo.log" 2>&1
) &
PID2=$!
sleep 10

log "GPU3 manipulation Gaussian PPO (Can)"
GPU=3 bash "${MANIP}/scripts/run_manip_gaussian_ppo.sh" \
  > "${PIPE_DIR}/manip_gaussian_ppo.log" 2>&1 &
PID3=$!

echo "${PID0} 0 baseline_plus_teacher" > "${PIPE_DIR}/post_eval_pids.txt"
echo "${PID1} 1 quant_qual" >> "${PIPE_DIR}/post_eval_pids.txt"
echo "${PID2} 2 g1_ppo" >> "${PIPE_DIR}/post_eval_pids.txt"
echo "${PID3} 3 gaussian_ppo" >> "${PIPE_DIR}/post_eval_pids.txt"
log "launched pids GPU0=${PID0} GPU1=${PID1} GPU2=${PID2} GPU3=${PID3}"

wait "$PID0" && log "OK GPU0 baseline_plus_teacher" || log "FAIL GPU0"
wait "$PID1" && log "OK GPU1 quant_qual" || log "FAIL GPU1"
wait "$PID2" && log "OK GPU2 g1_ppo" || log "FAIL GPU2"
wait "$PID3" && log "OK GPU3 gaussian_ppo" || log "FAIL GPU3"
log "post-eval 4gpu finished"
touch "${PIPE_DIR}/POST_EVAL_DONE"
