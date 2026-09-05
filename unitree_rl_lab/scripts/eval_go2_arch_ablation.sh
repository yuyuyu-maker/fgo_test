#!/usr/bin/env bash
# Step-sweep eval for official Unitree-Go2-Velocity architecture ablation.
# Protocol: play env, 256 envs, 10 episodes/env, steps 64/32/16/8/4/1, modes zero/random.
#
# Usage:
#   GPU=0 bash scripts/eval_go2_arch_ablation.sh
#   VARIANTS="reflow reward_aware" GPU=1 bash scripts/eval_go2_arch_ablation.sh
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
set -u

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
unset CUDA_VISIBLE_DEVICES || true
export TORCHDYNAMO_DISABLE=1

GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-256}"
STEPS="${STEPS:-64 32 16 8 4 1}"
PIPE_DIR="${ROOT}/logs/fpo/go2_arch_ablation"
mkdir -p "$PIPE_DIR"
STAMP="$(cat "${PIPE_DIR}/stamp.txt" 2>/dev/null || true)"
SUMMARY="${SUMMARY:-${PIPE_DIR}/step_sweep_summary.txt}"
OUT="${PIPE_DIR}/eval.out"

read -r -a VARIANT_LIST <<< "${VARIANTS:-baseline reflow reflow_teacher_kd reward_aware kd_only all_ideas_teacher_kd}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

exp_dir_for() {
  case "$1" in
    baseline) echo "${ROOT}/logs/fpo/unitree_go2_flat_flow" ;;
    reflow) echo "${ROOT}/logs/fpo/unitree_go2_flat_flow_reflow" ;;
    reflow_teacher_kd) echo "${ROOT}/logs/fpo/unitree_go2_reflow_teacher_kd" ;;
    reward_aware) echo "${ROOT}/logs/fpo/unitree_go2_reflow_reward_aware" ;;
    kd_only) echo "${ROOT}/logs/fpo/unitree_go2_kd_only" ;;
    all_ideas_teacher_kd) echo "${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher" ;;
    reflow_adaptive_lambda) echo "${ROOT}/logs/fpo/unitree_go2_reflow_adaptive_lambda" ;;
    reflow_random_x0) echo "${ROOT}/logs/fpo/unitree_go2_reflow_random_x0" ;;
    baseline_plus_teacher) echo "${ROOT}/logs/fpo/unitree_go2_baseline_plus_teacher" ;;
    *) return 1 ;;
  esac
}

resolve_ckpt() {
  local variant="$1"
  local exp
  exp="$(exp_dir_for "$variant")" || return 1
  local ckpt=""
  if [[ -n "$STAMP" ]]; then
    ckpt="$(ls -1d "${exp}/"*"arch_${STAMP}_${variant}"/model_1499.pt 2>/dev/null | tail -1 || true)"
  fi
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    ckpt="$(ls -1t "${exp}"/*/model_1499.pt 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    ckpt="$(ls -1t "${exp}"/*/model_*.pt 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    return 1
  fi
  echo "$ckpt"
}

: > "$SUMMARY"
log "Go2 arch ablation step-sweep GPU=${GPU} variants=[${VARIANT_LIST[*]}] stamp=${STAMP:-none}"
{
  echo "# Official Unitree-Go2-Velocity play-env step sweep"
  echo "# stamp=${STAMP:-none}  envs=${NUM_ENVS}  steps=${STEPS}  modes=zero/random"
  echo
} | tee -a "$SUMMARY"

for variant in "${VARIANT_LIST[@]}"; do
  ckpt="$(resolve_ckpt "$variant" || true)"
  eval_log="${PIPE_DIR}/eval_${variant}.log"
  echo "=== ${variant} ===" | tee -a "$SUMMARY" "$OUT"
  echo "checkpoint: ${ckpt:-MISSING}" | tee -a "$SUMMARY" "$OUT"

  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "SKIP: missing checkpoint" | tee -a "$SUMMARY" "$OUT"
    echo | tee -a "$SUMMARY" "$OUT"
    continue
  fi

  set +e
  # shellcheck disable=SC2086
  python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
    --task Unitree-Go2-Velocity \
    --headless \
    --device "cuda:${GPU}" \
    --num_envs "${NUM_ENVS}" \
    --eval_episodes 10 \
    --sampling_steps ${STEPS} \
    --eval_modes zero random \
    --fpo_variant "${variant}" \
    --model "${variant}=${ckpt}" \
    > "${eval_log}" 2>&1
  rc=$?

  if [[ "$rc" -ne 0 ]] || ! grep -q "SUMMARY TABLE" "${eval_log}"; then
    echo "FAILED rc=${rc}" | tee -a "$SUMMARY" "$OUT"
    tail -n 40 "${eval_log}" | tee -a "$SUMMARY" "$OUT"
    echo | tee -a "$SUMMARY" "$OUT"
    continue
  fi

  grep -A 20 "SUMMARY TABLE" "${eval_log}" | tee -a "$SUMMARY" "$OUT"
  echo | tee -a "$SUMMARY" "$OUT"
  cp -f "${eval_log}" "$(dirname "$ckpt")/step_sweep.log" 2>/dev/null || true
  grep -A 20 "SUMMARY TABLE" "${eval_log}" > "$(dirname "$ckpt")/step_sweep_summary.txt" 2>/dev/null || true
done

log "step-sweep finished. summary=${SUMMARY}"
echo "Summary: ${SUMMARY}"
