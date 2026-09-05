#!/usr/bin/env bash
# Resume missing 3-tier legs after prior failures.
#   GPU0  G1: FPO++ resume model_750 -> ours
#   GPU1  H1: ours only (FPO++ already finished)
#   GPU3  Spot: FPO++ -> ours
#   GPU2  leave BoxCleanup alone
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PIPE="${ROOT}/logs/fpo/gpu_queues"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$PIPE"
exec > >(tee -a "${PIPE}/3tier_resume_${STAMP}.out") 2>&1

# Always log to stderr so $(...) only captures paths.
log() { echo "[$(date '+%F %T')] $*" >&2; }

G1_TEACHER="/dev/shm/unitree_g1_ppo_gpu0/logs/rsl_rl/unitree_g1_29dof_velocity/2026-09-03_21-16-01_unitree_g1_ppo_2026-09-03_21-13-33/model_2999.pt"
H1_TEACHER="/dev/shm/unitree_h1_ppo_gpu1/logs/rsl_rl/unitree_h1_velocity/2026-09-03_23-14-20_unitree_h1_ppo_2026-09-03_22-58-32/model_2999.pt"
SPOT_TEACHER="/dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_flat/2026-09-03_23-13-07_spot_ppo_2026-09-03_23-12-19/model_1499.pt"
G1_FPO_CKPT="/dev/shm/unitree_g1_fpo_baseline_gpu0/logs/fpo/unitree_g1_29dof_fpo_baseline/2026-09-03_23-28-14_g1_fpo_baseline_resume_2026-09-03_23-27-25/model_750.pt"

for p in "$G1_TEACHER" "$H1_TEACHER" "$SPOT_TEACHER" "$G1_FPO_CKPT"; do
  [[ -f "$p" ]] || { log "ERROR missing: $p"; exit 1; }
done

log "3-tier resume stamp=${STAMP}"
: > "${PIPE}/3tier_resume_pids.txt"

# --- GPU0: G1 FPO++ resume -> ours ---
(
  set -eo pipefail
  log "GPU0 G1 FPO++ resume from model_750 (remaining ~750 iters -> 1500)"
  GPU=0 FPO_VARIANT=baseline EXPERIMENT_NAME=unitree_g1_29dof_fpo_baseline \
    RESUME=1 CHECKPOINT="$G1_FPO_CKPT" MAX_ITERS=750 \
    RUN_NAME="g1_fpo_baseline_resume750_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_baseline_gpu0 \
    bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu0_g1_fpo_baseline.log" 2>&1
  log "GPU0 G1 ours (all_ideas_teacher_kd)"
  GPU=0 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=unitree_g1_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$G1_TEACHER" RUN_NAME="g1_all_ideas_${STAMP}" \
    WORKDIR=/dev/shm/unitree_g1_fpo_ours_gpu0 \
    bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu0_g1_ours.log" 2>&1
  log "OK GPU0 G1 FPO++ + ours"
) &
echo "$! gpu0_g1" >> "${PIPE}/3tier_resume_pids.txt"

sleep 15

# --- GPU1: H1 ours only ---
(
  set -eo pipefail
  log "GPU1 H1 ours (FPO++ already done)"
  GPU=1 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=unitree_h1_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$H1_TEACHER" RUN_NAME="h1_all_ideas_${STAMP}" \
    WORKDIR=/dev/shm/unitree_h1_fpo_ours_gpu1 \
    bash "${ROOT}/scripts/run_h1_fpo.sh" > "${PIPE}/gpu1_h1_ours.log" 2>&1
  log "OK GPU1 H1 ours"
) &
echo "$! gpu1_h1" >> "${PIPE}/3tier_resume_pids.txt"

sleep 15

# --- GPU3: Spot FPO++ -> ours ---
(
  set -eo pipefail
  log "GPU3 Spot FPO++ baseline"
  GPU=3 FPO_VARIANT=baseline EXPERIMENT_NAME=spot_flat_flow \
    RUN_NAME="spot_fpo_baseline_${STAMP}" WORKDIR=/dev/shm/spot_fpo_baseline_gpu3 \
    bash "${ROOT}/scripts/run_spot_fpo.sh" > "${PIPE}/gpu3_spot_fpo_baseline.log" 2>&1
  log "GPU3 Spot ours"
  GPU=3 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=spot_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$SPOT_TEACHER" RUN_NAME="spot_all_ideas_${STAMP}" \
    WORKDIR=/dev/shm/spot_fpo_ours_gpu3 \
    bash "${ROOT}/scripts/run_spot_fpo.sh" > "${PIPE}/gpu3_spot_ours.log" 2>&1
  log "OK GPU3 Spot FPO++ + ours"
) &
echo "$! gpu3_spot" >> "${PIPE}/3tier_resume_pids.txt"

log "resume queues armed: $(tr '\n' ' ' < "${PIPE}/3tier_resume_pids.txt")"
wait
log "all resume queues done"
touch "${PIPE}/TIER3_RESUME_DONE"
