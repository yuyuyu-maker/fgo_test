#!/usr/bin/env bash
# Quantitative + qualitative Go2 analysis after architecture eval.
# TB dump (no Kit) + pinned-command tracking plots for PPO / FPO++ / full / 1-step.
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
set -u
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
unset CUDA_VISIBLE_DEVICES || true

GPU="${GPU:-1}"
OUT="${ROOT}/logs/fpo/go2_arch_ablation/quant_qual"
mkdir -p "$OUT"
LOG="${OUT}/quant_qual.log"

PPO_CKPT="${PPO_CKPT:-${ROOT}/logs/rsl_rl/unitree_go2_velocity/2026-09-02_17-20-40_unitree_go2_ppo/model_2999.pt}"
FPO_PP_CKPT="${FPO_PP_CKPT:-${ROOT}/logs/fpo/unitree_go2_flat_flow/2026-09-02_20-16-21_fpo_baseline_unitree_mdp/model_1499.pt}"
FULL_CKPT="${FULL_CKPT:-${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300/model_1499.pt}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

python -u "${ROOT}/scripts/analyze_go2_tb.py" 2>&1 | tee -a "$LOG"

run_track() {
  local kind="$1"
  local ckpt="$2"
  local sub="$3"
  shift 3
  local dest="${OUT}/${sub}"
  mkdir -p "$dest"
  log "TRACK ${sub} ckpt=${ckpt}"
  if [[ ! -f "$ckpt" ]]; then
    log "SKIP missing ${ckpt}"
    return 0
  fi
  python -u "${ROOT}/scripts/eval_go2_tracking.py" \
    --kind "$kind" \
    --task Unitree-Go2-Velocity \
    --headless \
    --device "cuda:${GPU}" \
    --num_envs 4 \
    --steps 500 \
    --checkpoint "$ckpt" \
    --out_dir "$dest" \
    --title "$sub" \
    "$@" >>"$LOG" 2>&1 || log "WARN tracking failed: ${sub}"
}

run_track ppo "$PPO_CKPT" ppo
run_track fpo "$FPO_PP_CKPT" fpo_plusplus_s64 --fpo_variant baseline --flow-sampling-steps 64
run_track fpo "$FPO_PP_CKPT" fpo_plusplus_s1 --fpo_variant baseline --flow-sampling-steps 1
run_track fpo "$FULL_CKPT" full_s64 --fpo_variant all_ideas_teacher_kd --flow-sampling-steps 64
run_track fpo "$FULL_CKPT" full_s1 --fpo_variant all_ideas_teacher_kd --flow-sampling-steps 1

REFLOW_CKPT="$(ls -1t "${ROOT}"/logs/fpo/unitree_go2_flat_flow_reflow/*/model_1499.pt 2>/dev/null | head -1 || true)"
RA_CKPT="$(ls -1t "${ROOT}"/logs/fpo/unitree_go2_reflow_reward_aware/*/model_1499.pt 2>/dev/null | head -1 || true)"
if [[ -n "$REFLOW_CKPT" ]]; then
  run_track fpo "$REFLOW_CKPT" reflow_s64 --fpo_variant reflow --flow-sampling-steps 64
  run_track fpo "$REFLOW_CKPT" reflow_s1 --fpo_variant reflow --flow-sampling-steps 1
fi
if [[ -n "$RA_CKPT" ]]; then
  run_track fpo "$RA_CKPT" reward_aware_s64 --fpo_variant reward_aware --flow-sampling-steps 64
  run_track fpo "$RA_CKPT" reward_aware_s1 --fpo_variant reward_aware --flow-sampling-steps 1
fi

# Optional: latest kd_only / reflow_teacher_kd at 1 and 64 if eval already produced ckpts.
KD_CKPT="$(ls -1t "${ROOT}"/logs/fpo/unitree_go2_kd_only/*/model_1499.pt 2>/dev/null | head -1 || true)"
RTK_CKPT="$(ls -1t "${ROOT}"/logs/fpo/unitree_go2_reflow_teacher_kd/*/model_1499.pt 2>/dev/null | head -1 || true)"
if [[ -n "$KD_CKPT" ]]; then
  run_track fpo "$KD_CKPT" kd_only_s64 --fpo_variant kd_only --flow-sampling-steps 64
  run_track fpo "$KD_CKPT" kd_only_s1 --fpo_variant kd_only --flow-sampling-steps 1
fi
if [[ -n "$RTK_CKPT" ]]; then
  run_track fpo "$RTK_CKPT" reflow_teacher_kd_s64 --fpo_variant reflow_teacher_kd --flow-sampling-steps 64
  run_track fpo "$RTK_CKPT" reflow_teacher_kd_s1 --fpo_variant reflow_teacher_kd --flow-sampling-steps 1
fi

{
  echo "# tracking summaries"
  find "$OUT" -name tracking_summary.txt -print -exec cat {} \;
} | tee "${OUT}/tracking_all.txt" | tee -a "$LOG"

log "quant/qual done -> ${OUT}"
