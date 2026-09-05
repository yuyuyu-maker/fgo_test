#!/usr/bin/env bash
# GPU2: quick Can eval (PPO + Ours PPO-teacher) — 200 ep, no video (bosfs-safe).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIP="/workspace/fgo_test/manipulation_experiments"
PIPE="${ROOT}/logs/fpo/gpu_queues"
mkdir -p "$PIPE"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="${MANIP}/logs/eval_can_paper_${STAMP}"
mkdir -p "$OUT"
LOG="${PIPE}/eval_can_gpu2_${STAMP}.out"
exec > >(tee -a "$LOG") 2>&1

GPU="${GPU:-2}"
NUM_ENVS="${NUM_ENVS:-16}"   # keep light to avoid OOM / disconnect pain
NUM_EPISODES="${NUM_EPISODES:-100}"

PPO_CKPT="${MANIP}/runs/gaussian_ppo_can_gpu2_2026-09-03_21-14-22/checkpoints/step_4800000"
OURS_CKPT="${MANIP}/runs/flow_fpo_can_all_ideas_teacher_kd_rlteacher_gpu2_2026-09-05_10-57-39/checkpoints/step_4800000"
# optional older FPO++ baseline if present
BASE_CKPT="$(ls -d ${MANIP}/runs/flow_fpo_can_baseline_gpu0_*/checkpoints/step_* 2>/dev/null | sort -V | tail -1 || true)"

log() { echo "[$(date '+%F %T')] $*"; }

eval_one() {
  local name="$1" ckpt="$2"
  if [[ ! -d "${ckpt}/policy" ]]; then
    log "SKIP ${name}: missing ${ckpt}/policy"
    return 0
  fi
  log "eval ${name} ckpt=${ckpt} envs=${NUM_ENVS} eps=${NUM_EPISODES}"
  cd "$MANIP"
  set +u; source source_env.sh; set -uo pipefail
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export TMPDIR="${TMPDIR:-/dev/shm/_can_eval_tmp}"; mkdir -p "$TMPDIR"
  export MUJOCO_GL=egl
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
    python eval_checkpoint.py \
      --local_checkpoint_path "${ckpt}" \
      --eval_env Can \
      --eval_num_episodes "${NUM_EPISODES}" \
      --eval-num-envs "${NUM_ENVS}" \
      --load-ema True \
      --wandb_enable False \
      --zero-sampling True \
      --save_video False \
      --experiment "eval_can_${name}_${STAMP}" \
      > "${OUT}/${name}.log" 2>&1 || log "FAIL ${name}"
  rg -n 'Success Rate|success_rate|SR:' "${OUT}/${name}.log" | tail -5 || tail -20 "${OUT}/${name}.log"
}

log "Can eval GPU${GPU} → ${OUT}"
echo "$$" > "${PIPE}/eval_can_gpu2.pid"

eval_one ppo "${PPO_CKPT}"
eval_one ours_ppoteacher "${OURS_CKPT}"
[[ -n "${BASE_CKPT}" ]] && eval_one fpo_baseline "${BASE_CKPT}"

log "SUMMARY"
for f in "${OUT}"/*.log; do
  echo "==== $(basename "$f") ===="
  rg 'Success Rate|success_rate|SR:|episodes' "$f" | tail -8 || true
done
echo "CAN_EVAL_DONE $(date -Iseconds) ${OUT}" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
log "DONE"
