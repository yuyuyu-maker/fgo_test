#!/usr/bin/env bash
# GPU3 Spot chain (OOM-safe, disconnect-tolerant):
#   wait live kd_only → Ours → play-fill FPO++/kd/Ours (train-env fallback).
# Spot budget stays 16384×1500; only one Kit job on this GPU at a time.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE="${ROOT}/logs/fpo/gpu_queues"
mkdir -p "$PIPE"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG="${PIPE}/queue_spot_gpu3_${STAMP}.out"
exec > >(tee -a "$LOG") 2>&1

GPU=3
SPOT_PPO="/dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_flat/2026-09-03_23-13-07_spot_ppo_2026-09-03_23-12-19/model_1499.pt"
SPOT_FPO="/dev/shm/spot_fpo_baseline_gpu3/logs/fpo/spot_flat_flow/2026-09-04_10-13-02_spot_fpo_baseline_2026-09-04_10-11-50/model_1499.pt"
KD_RUN="/dev/shm/spot_fpo_kd_only_gpu3/logs/fpo/spot_kd_only/2026-09-05_10-24-05_spot_kd_only_fpo_default_2026-09-05_10-23-19"
OURS_WORKDIR="/dev/shm/spot_fpo_ours_gpu3"

log() { echo "[$(date '+%F %T')] $*"; }

wait_gpu_idle() {
  local thr_mib="${1:-3000}"
  while true; do
    local used
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    # also require no spot train.py
    if [[ "$used" -lt "$thr_mib" ]] && ! pgrep -af 'fpo/train.py' | grep -q "Isaac-Velocity-Flat-Spot-v0.*cuda:${GPU}"; then
      return 0
    fi
    sleep 60
  done
}

play_spot() {
  local name="$1" variant="$2" ckpt="$3"
  [[ -f "$ckpt" ]] || { log "SKIP play ${name}: missing ${ckpt}"; return 0; }
  local out="${PIPE}/spot_play_${STAMP}"
  mkdir -p "$out"
  log "play ${name} (${variant}) → ${out}/${name}.log"
  set +u
  source /workspace/fgo_test/isaaclab_experiments/source_env.sh
  set -uo pipefail
  export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
  export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
  unset CUDA_VISIBLE_DEVICES || true
  # Prefer Play-v0; eval_sampling_steps falls back to env_cfg if no play entry.
  python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
    --task Isaac-Velocity-Flat-Spot-Play-v0 --headless --device "cuda:${GPU}" \
    --num_envs 256 --eval_episodes 10 --sampling_steps 64 1 --eval_modes zero \
    --fpo_variant "$variant" --model "${name}=${ckpt}" \
    > "${out}/${name}.log" 2>&1 || \
  python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
    --task Isaac-Velocity-Flat-Spot-v0 --headless --device "cuda:${GPU}" \
    --num_envs 256 --eval_episodes 10 --sampling_steps 64 1 --eval_modes zero \
    --fpo_variant "$variant" --model "${name}=${ckpt}" \
    > "${out}/${name}.log" 2>&1 || log "FAIL play ${name}"
}

log "Spot queue GPU3 stamp=${STAMP}"
echo "$$" > "${PIPE}/queue_spot_gpu3.pid"

# 1) Wait for live kd_only (pid 302035 or any Spot kd on GPU3)
if pgrep -af 'fpo/train.py' | grep -q 'Isaac-Velocity-Flat-Spot-v0.*kd_only'; then
  log "waiting for Spot kd_only to finish..."
  while pgrep -af 'fpo/train.py' | grep -q 'Isaac-Velocity-Flat-Spot-v0.*kd_only'; do
    latest=$(ls -1 "${KD_RUN}"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    log "  still training; latest=${latest:-none}"
    sleep 180
  done
  log "Spot kd_only exited"
else
  log "no live Spot kd_only; will use existing ckpt if present"
fi

KD_CKPT="$(ls -1 "${KD_RUN}"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "${KD_CKPT}" ]]; then
  KD_CKPT="$(ls -1t /dev/shm/spot_fpo_kd_only_gpu3/logs/fpo/spot_kd_only/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
log "kd ckpt=${KD_CKPT:-MISSING}"

wait_gpu_idle 4000
log "GPU3 idle → Spot Ours (16384×1500, exclusive)"

GPU=3 FPO_VARIANT=all_ideas_teacher_kd NUM_ENVS=16384 MAX_ITERS=1500 \
  TEACHER_CHECKPOINT="${SPOT_PPO}" \
  EXPERIMENT_NAME=spot_all_ideas_ppo_teacher \
  RUN_NAME="spot_ours_${STAMP}" \
  WORKDIR="${OURS_WORKDIR}" \
  bash "${ROOT}/scripts/run_spot_fpo.sh" \
  > "${PIPE}/gpu3_spot_ours_${STAMP}.log" 2>&1
log "Spot Ours train exit=$?"

OURS_CKPT="$(ls -1t ${OURS_WORKDIR}/logs/fpo/*/model_*.pt 2>/dev/null | head -1 || true)"
[[ -z "${OURS_CKPT}" ]] && OURS_CKPT="$(ls -1t ${ROOT}/logs/fpo/spot_all_ideas_ppo_teacher/*/model_*.pt 2>/dev/null | head -1 || true)"

wait_gpu_idle 3000
play_spot Spot_FPO_pp baseline "${SPOT_FPO}"
play_spot Spot_kd kd_only "${KD_CKPT}"
play_spot Spot_Ours all_ideas_teacher_kd "${OURS_CKPT}"

log "Spot queue DONE"
echo "SPOT_QUEUE_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
