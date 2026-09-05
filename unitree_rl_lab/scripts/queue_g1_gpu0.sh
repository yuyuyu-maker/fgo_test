#!/usr/bin/env bash
# GPU0 G1 chain @ official budget 4096×50000:
#   wait live PPO 50k → FPO++ → kd_only → Ours → play 64/1
# Exclusive GPU0 (H1 moved off this card).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE="${ROOT}/logs/fpo/gpu_queues"
mkdir -p "$PIPE"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG="${PIPE}/queue_g1_gpu0_${STAMP}.out"
exec > >(tee -a "$LOG") 2>&1

GPU=0
PPO_RUN="/dev/shm/unitree_g1_ppo_official_gpu1/logs/rsl_rl/unitree_g1_29dof_velocity/2026-09-05_10-08-49_g1_ppo_official_50k_2026-09-05_10-07-27"
BASE_WORKDIR="/dev/shm/unitree_g1_fpo_baseline_50k_gpu0"
KD_WORKDIR="/dev/shm/unitree_g1_fpo_kd_50k_gpu0"
OURS_WORKDIR="/dev/shm/unitree_g1_fpo_ours_50k_gpu0"

log() { echo "[$(date '+%F %T')] $*"; }

wait_gpu_idle() {
  local thr_mib="${1:-4000}"
  while true; do
    local used
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" -lt "$thr_mib" ]] && ! pgrep -af 'train.py' | grep -q "Unitree-G1-29dof-Velocity.*cuda:${GPU}"; then
      return 0
    fi
    sleep 90
  done
}

play_g1() {
  local name="$1" variant="$2" ckpt="$3"
  [[ -f "$ckpt" ]] || { log "SKIP play ${name}"; return 0; }
  local out="${PIPE}/g1_play_${STAMP}"; mkdir -p "$out"
  log "play ${name}"
  set +u; source /workspace/fgo_test/isaaclab_experiments/source_env.sh; set -uo pipefail
  export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
  unset CUDA_VISIBLE_DEVICES || true
  python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
    --task Unitree-G1-29dof-Velocity --headless --device "cuda:${GPU}" \
    --num_envs 256 --eval_episodes 10 --sampling_steps 64 1 --eval_modes zero \
    --fpo_variant "$variant" --model "${name}=${ckpt}" \
    > "${out}/${name}.log" 2>&1 || log "FAIL play ${name}"
}

log "G1 queue GPU0 stamp=${STAMP}"
echo "$$" > "${PIPE}/queue_g1_gpu0.pid"

if pgrep -af 'rsl_rl/train.py' | grep -q 'Unitree-G1-29dof-Velocity'; then
  log "waiting for G1 PPO 50k..."
  while pgrep -af 'rsl_rl/train.py' | grep -q 'Unitree-G1-29dof-Velocity'; do
    latest=$(ls -1 "${PPO_RUN}"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    log "  PPO latest=${latest:-none}"
    sleep 300
  done
  log "G1 PPO exited"
fi

PPO_CKPT="$(ls -1 "${PPO_RUN}"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
log "G1 PPO ckpt=${PPO_CKPT:-MISSING}"
[[ -n "${PPO_CKPT}" ]] || { log "ABORT: no PPO ckpt"; exit 1; }

wait_gpu_idle 4000
log "G1 FPO++ baseline 4096×50000"
GPU=0 FPO_VARIANT=baseline NUM_ENVS=4096 MAX_ITERS=50000 \
  EXPERIMENT_NAME=unitree_g1_29dof_fpo_baseline \
  RUN_NAME="g1_baseline_50k_${STAMP}" \
  WORKDIR="${BASE_WORKDIR}" \
  bash "${ROOT}/scripts/run_g1_fpo.sh" \
  > "${PIPE}/gpu0_g1_baseline_${STAMP}.log" 2>&1
log "G1 baseline exit=$?"

BASE_CKPT="$(ls -1t ${BASE_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 4000
log "G1 kd_only"
GPU=0 FPO_VARIANT=kd_only NUM_ENVS=4096 MAX_ITERS=50000 \
  TEACHER_CHECKPOINT="${PPO_CKPT}" \
  EXPERIMENT_NAME=unitree_g1_kd_only \
  RUN_NAME="g1_kd_50k_${STAMP}" \
  WORKDIR="${KD_WORKDIR}" \
  bash "${ROOT}/scripts/run_g1_fpo.sh" \
  > "${PIPE}/gpu0_g1_kd_${STAMP}.log" 2>&1
log "G1 kd exit=$?"

KD_CKPT="$(ls -1t ${KD_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 4000
log "G1 Ours"
GPU=0 FPO_VARIANT=all_ideas_teacher_kd NUM_ENVS=4096 MAX_ITERS=50000 \
  TEACHER_CHECKPOINT="${PPO_CKPT}" \
  EXPERIMENT_NAME=unitree_g1_all_ideas_ppo_teacher \
  RUN_NAME="g1_ours_50k_${STAMP}" \
  WORKDIR="${OURS_WORKDIR}" \
  bash "${ROOT}/scripts/run_g1_fpo.sh" \
  > "${PIPE}/gpu0_g1_ours_${STAMP}.log" 2>&1
log "G1 Ours exit=$?"

OURS_CKPT="$(ls -1t ${OURS_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 3000
set +u; source /workspace/fgo_test/isaaclab_experiments/source_env.sh; set -uo pipefail
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}" OMNI_KIT_ACCEPT_EULA=YES
OUTP="${PIPE}/g1_play_${STAMP}"; mkdir -p "$OUTP"
python -u "${ROOT}/scripts/eval_ppo_play.py" \
  --task Unitree-G1-29dof-Velocity --checkpoint "${PPO_CKPT}" \
  --headless --device "cuda:${GPU}" --num_envs 256 --eval_episodes 10 \
  > "${OUTP}/G1_PPO.log" 2>&1 || log "FAIL G1 PPO play"
play_g1 G1_FPO_pp baseline "${BASE_CKPT}"
play_g1 G1_kd kd_only "${KD_CKPT}"
play_g1 G1_Ours all_ideas_teacher_kd "${OURS_CKPT}"

log "G1 queue DONE"
echo "G1_QUEUE_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
