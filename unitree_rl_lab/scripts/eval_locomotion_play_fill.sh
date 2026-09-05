#!/usr/bin/env bash
# Fill locomotion main-table play rewards for finished checkpoints (non-G1-train).
# Usage: GPU=2 bash scripts/eval_locomotion_play_fill.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
set -uo pipefail

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
unset CUDA_VISIBLE_DEVICES || true
export TORCHDYNAMO_DISABLE=1

GPU="${GPU:-2}"
NUM_ENVS="${NUM_ENVS:-256}"
OUT="${ROOT}/logs/fpo/play_fill_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$OUT"
SUMMARY="${OUT}/SUMMARY.jsonl"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*"; }

run_ppo() {
  local name="$1" task="$2" ckpt="$3"
  if [[ ! -f "$ckpt" ]]; then
    log "SKIP PPO ${name}: missing ${ckpt}"
    return 0
  fi
  log "PPO play ${name}"
  local logf="${OUT}/${name}_ppo.log"
  if python -u "${ROOT}/scripts/eval_ppo_play.py" \
      --task "$task" --checkpoint "$ckpt" \
      --headless --device "cuda:${GPU}" \
      --num_envs "$NUM_ENVS" --eval_episodes 10 \
      2>&1 | tee "$logf"; then
    grep 'SUMMARY_JSON ' "$logf" | sed 's/.*SUMMARY_JSON //' >> "$SUMMARY" || true
  else
    log "FAIL PPO ${name}"
  fi
}

run_fpo() {
  local name="$1" task="$2" variant="$3" ckpt="$4"
  if [[ ! -f "$ckpt" ]]; then
    log "SKIP FPO ${name}: missing ${ckpt}"
    return 0
  fi
  log "FPO play ${name} (${variant}) steps=64,1"
  local logf="${OUT}/${name}_${variant}.log"
  if python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
      --task "$task" --headless --device "cuda:${GPU}" \
      --num_envs "$NUM_ENVS" --eval_episodes 10 \
      --sampling_steps 64 1 --eval_modes zero \
      --fpo_variant "$variant" \
      --model "${name}=${ckpt}" \
      2>&1 | tee "$logf"; then
    # extract zero-mode 64/1 if present
    python3 - <<PY >>"$SUMMARY"
import json, re
text=open("$logf").read()
# lines like: [zero] steps= 64 reward=  21.29
rows={}
for m in re.finditer(r"steps=\s*(\d+)\s+reward=\s*([-\d.]+)", text):
    rows[int(m.group(1))]=float(m.group(2))
print(json.dumps({"task":"$task","algo":"$variant","label":"$name","checkpoint":"$ckpt","rewards":rows}))
PY
  else
    log "FAIL FPO ${name}"
  fi
}

SPOT_PPO="/dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_flat/2026-09-03_23-13-07_spot_ppo_2026-09-03_23-12-19/model_1499.pt"
H1_PPO="/dev/shm/unitree_h1_ppo_gpu1/logs/rsl_rl/unitree_h1_velocity/2026-09-03_23-14-20_unitree_h1_ppo_2026-09-03_22-58-32/model_2999.pt"
G1_PPO="${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity/2026-09-04_11-15-17_g1_ppo_extend_curriculum_2026-09-04_11-14-23/model_9998.pt"
H1_FPO="${ROOT}/logs/fpo/unitree_h1_flat_flow/2026-09-04_01-51-15_h1_fpo_baseline_2026-09-03_23-27-25/model_1800.pt"
G1_FPO="/dev/shm/unitree_g1_fpo_baseline_gpu1/logs/fpo/unitree_g1_29dof_fpo_baseline/2026-09-04_16-47-54_g1_fpo_baseline_resume1150_2026-09-04_12-17-34/model_1499.pt"
# Spot FPO++ if already finished
SPOT_FPO="$(ls -1t /dev/shm/spot_fpo_baseline_gpu3/logs/fpo/spot_flat_flow/*/model_1499.pt 2>/dev/null | head -1 || true)"

log "OUT=${OUT}"

run_ppo Spot_PPO "Isaac-Velocity-Flat-Spot-Play-v0" "$SPOT_PPO" || \
  run_ppo Spot_PPO "Isaac-Velocity-Flat-Spot-v0" "$SPOT_PPO"

run_ppo H1_PPO "Unitree-H1-Velocity" "$H1_PPO"
run_ppo G1_PPO "Unitree-G1-29dof-Velocity" "$G1_PPO"

run_fpo H1_FPO_pp "Unitree-H1-Velocity" baseline "$H1_FPO"
run_fpo G1_FPO_pp "Unitree-G1-29dof-Velocity" baseline "$G1_FPO"

if [[ -n "${SPOT_FPO}" ]]; then
  run_fpo Spot_FPO_pp "Isaac-Velocity-Flat-Spot-Play-v0" baseline "$SPOT_FPO" || \
    run_fpo Spot_FPO_pp "Isaac-Velocity-Flat-Spot-v0" baseline "$SPOT_FPO"
fi

log "DONE play fill → ${SUMMARY}"
echo "PLAY_FILL_DONE $(date -Iseconds) ${OUT}" >> "${ROOT}/logs/fpo/gpu_queues/KEEP_BUSY_EVENTS.txt"
