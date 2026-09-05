#!/usr/bin/env bash
# GPU1 H1 chain @ new budget 8192×2000 (exclusive card, resume-friendly):
#   resume kd_only from model_1000 → FPO++ baseline (no teacher) → Ours → play 64/1
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE="${ROOT}/logs/fpo/gpu_queues"
mkdir -p "$PIPE"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG="${PIPE}/queue_h1_gpu1_${STAMP}.out"
exec > >(tee -a "$LOG") 2>&1

GPU=1
H1_PPO="/dev/shm/unitree_h1_ppo_aligned_gpu0/logs/rsl_rl/unitree_h1_velocity/2026-09-05_10-40-24_h1_ppo_aligned_fpo_budget_2026-09-05_10-39-20/model_1999.pt"
KD_CKPT_SRC="/dev/shm/unitree_h1_fpo_kd_only_8192_gpu0/logs/fpo/unitree_h1_kd_only/2026-09-05_11-33-29_h1_kd_only_8192_2026-09-05_10-47-26/model_1000.pt"
KD_WORKDIR="/dev/shm/unitree_h1_fpo_kd_only_8192_gpu1"
BASE_WORKDIR="/dev/shm/unitree_h1_fpo_baseline_8192_gpu1"
OURS_WORKDIR="/dev/shm/unitree_h1_fpo_ours_8192_gpu1"

log() { echo "[$(date '+%F %T')] $*"; }

wait_gpu_idle() {
  local thr_mib="${1:-4000}"
  while true; do
    local used
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" -lt "$thr_mib" ]] && ! pgrep -af 'fpo/train.py' | grep -q "Unitree-H1-Velocity.*cuda:${GPU}"; then
      return 0
    fi
    sleep 45
  done
}

play_h1() {
  local name="$1" variant="$2" ckpt="$3"
  [[ -f "$ckpt" ]] || { log "SKIP play ${name}"; return 0; }
  local out="${PIPE}/h1_play_${STAMP}"
  mkdir -p "$out"
  log "play ${name} → ${out}/${name}.log"
  set +u; source /workspace/fgo_test/isaaclab_experiments/source_env.sh; set -uo pipefail
  export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
  export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
  unset CUDA_VISIBLE_DEVICES || true
  python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
    --task Unitree-H1-Velocity --headless --device "cuda:${GPU}" \
    --num_envs 256 --eval_episodes 10 --sampling_steps 64 1 --eval_modes zero \
    --fpo_variant "$variant" --model "${name}=${ckpt}" \
    > "${out}/${name}.log" 2>&1 || log "FAIL play ${name}"
}

log "H1 queue GPU1 stamp=${STAMP}"
echo "$$" > "${PIPE}/queue_h1_gpu1.pid"

# Stop H1 on GPU0 if still fighting G1 PPO (resume from synced ckpt).
if pgrep -af 'fpo/train.py' | grep -q 'Unitree-H1-Velocity.*cuda:0'; then
  log "stopping H1 on GPU0 to free card for G1; will resume on GPU1"
  pkill -f 'Unitree-H1-Velocity.*cuda:0' || true
  sleep 15
fi

# Refresh latest kd ckpt after stop (may have advanced past 1000).
LATEST_KD="$(ls -1 /dev/shm/unitree_h1_fpo_kd_only_8192_gpu0/logs/fpo/unitree_h1_kd_only/*/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
[[ -n "${LATEST_KD}" ]] && KD_CKPT_SRC="${LATEST_KD}"
log "resume kd from ${KD_CKPT_SRC}"

wait_gpu_idle 12000
# --- kd_only resume ---
if [[ -f "${KD_CKPT_SRC}" ]]; then
  # remaining iters: max 2000 absolute; resume continues counting from ckpt iter
  GPU=1 FPO_VARIANT=kd_only NUM_ENVS=8192 MAX_ITERS=2000 \
    TEACHER_CHECKPOINT="${H1_PPO}" \
    RESUME=1 CHECKPOINT="${KD_CKPT_SRC}" \
    EXPERIMENT_NAME=unitree_h1_kd_only \
    RUN_NAME="h1_kd_resume_${STAMP}" \
    WORKDIR="${KD_WORKDIR}" \
    bash "${ROOT}/scripts/run_h1_fpo.sh" \
    > "${PIPE}/gpu1_h1_kd_${STAMP}.log" 2>&1
  log "H1 kd exit=$?"
else
  log "WARN no kd ckpt; training kd from scratch"
  GPU=1 FPO_VARIANT=kd_only NUM_ENVS=8192 MAX_ITERS=2000 \
    TEACHER_CHECKPOINT="${H1_PPO}" \
    EXPERIMENT_NAME=unitree_h1_kd_only \
    RUN_NAME="h1_kd_${STAMP}" \
    WORKDIR="${KD_WORKDIR}" \
    bash "${ROOT}/scripts/run_h1_fpo.sh" \
    > "${PIPE}/gpu1_h1_kd_${STAMP}.log" 2>&1
fi

KD_FINAL="$(ls -1t ${KD_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"
[[ -z "${KD_FINAL}" ]] && KD_FINAL="$(ls -1t ${ROOT}/logs/fpo/unitree_h1_kd_only/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 4000
# --- FPO++ baseline: NEVER pass TEACHER (would be loaded as flow actor) ---
log "H1 FPO++ baseline (no teacher)"
GPU=1 FPO_VARIANT=baseline NUM_ENVS=8192 MAX_ITERS=2000 \
  EXPERIMENT_NAME=unitree_h1_flat_flow \
  RUN_NAME="h1_baseline_${STAMP}" \
  WORKDIR="${BASE_WORKDIR}" \
  bash "${ROOT}/scripts/run_h1_fpo.sh" \
  > "${PIPE}/gpu1_h1_baseline_${STAMP}.log" 2>&1
log "H1 baseline exit=$?"

BASE_FINAL="$(ls -1t ${BASE_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"
[[ -z "${BASE_FINAL}" ]] && BASE_FINAL="$(ls -1t ${ROOT}/logs/fpo/unitree_h1_flat_flow/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 4000
log "H1 Ours"
GPU=1 FPO_VARIANT=all_ideas_teacher_kd NUM_ENVS=8192 MAX_ITERS=2000 \
  TEACHER_CHECKPOINT="${H1_PPO}" \
  EXPERIMENT_NAME=unitree_h1_all_ideas_ppo_teacher \
  RUN_NAME="h1_ours_${STAMP}" \
  WORKDIR="${OURS_WORKDIR}" \
  bash "${ROOT}/scripts/run_h1_fpo.sh" \
  > "${PIPE}/gpu1_h1_ours_${STAMP}.log" 2>&1
log "H1 Ours exit=$?"

OURS_FINAL="$(ls -1t ${OURS_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 3000
# PPO play with aligned teacher
set +u; source /workspace/fgo_test/isaaclab_experiments/source_env.sh; set -uo pipefail
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}" OMNI_KIT_ACCEPT_EULA=YES
OUTP="${PIPE}/h1_play_${STAMP}"; mkdir -p "$OUTP"
python -u "${ROOT}/scripts/eval_ppo_play.py" \
  --task Unitree-H1-Velocity --checkpoint "${H1_PPO}" \
  --headless --device "cuda:${GPU}" --num_envs 256 --eval_episodes 10 \
  > "${OUTP}/H1_PPO.log" 2>&1 || log "FAIL H1 PPO play"

play_h1 H1_FPO_pp baseline "${BASE_FINAL}"
play_h1 H1_kd kd_only "${KD_FINAL}"
play_h1 H1_Ours all_ideas_teacher_kd "${OURS_FINAL}"

log "H1 queue DONE"
echo "H1_QUEUE_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
