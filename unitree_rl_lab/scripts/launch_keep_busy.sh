#!/usr/bin/env bash
# Keep 4 GPUs busy for locomotion 4-tier protocol.
# Order per task: PPO → FPO++ → FPO++ + teacher (kd_only, FROM SCRATCH) → Ours
# Does not kill live trains; freezes old Spot waiter so ours is not launched without kd_only.
#
#   nohup bash scripts/launch_keep_busy.sh >> logs/fpo/gpu_queues/keep_busy.out 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PIPE="${ROOT}/logs/fpo/gpu_queues"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$PIPE"
LOG="${PIPE}/keep_busy_${STAMP}.out"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

G1_TEACHER_OLD="/dev/shm/unitree_g1_ppo_gpu0/logs/rsl_rl/unitree_g1_29dof_velocity/2026-09-03_21-16-01_unitree_g1_ppo_2026-09-03_21-13-33/model_2999.pt"
H1_TEACHER="/dev/shm/unitree_h1_ppo_gpu1/logs/rsl_rl/unitree_h1_velocity/2026-09-03_23-14-20_unitree_h1_ppo_2026-09-03_22-58-32/model_2999.pt"
SPOT_TEACHER="/dev/shm/spot_ppo_gpu3/logs/rsl_rl/spot_flat/2026-09-03_23-13-07_spot_ppo_2026-09-03_23-12-19/model_1499.pt"
G1_FPO_CKPT="$(ls -1t /dev/shm/unitree_g1_fpo_baseline_gpu0/logs/fpo/unitree_g1_29dof_fpo_baseline/*/model_*.pt 2>/dev/null | head -1 || true)"

gpu_mem_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

gpu_idle() {
  local m
  m="$(gpu_mem_mib "$1")"
  # treat < 3GiB as free enough to launch a new Isaac job
  [[ "${m:-99999}" -lt 3000 ]]
}

wait_gpu_idle() {
  local gpu="$1" name="$2"
  log "wait GPU${gpu} idle for ${name} (mem<3GiB)"
  while ! gpu_idle "$gpu"; do sleep 60; done
  sleep 15
  log "GPU${gpu} free → ${name}"
}

latest_ckpt() {
  ls -1t "$@" 2>/dev/null | head -1 || true
}

# --- Freeze old Spot 3tier waiter shells (keep python Spot FPO++ alive) ---
freeze_old_spot_waiter() {
  local pids pid
  pids="$(pgrep -f 'launch_3tier_resume.sh' || true)"
  if [[ -z "$pids" ]]; then
    log "no launch_3tier_resume to freeze"
    return 0
  fi
  # Only STOP the 3tier bash waiters (not python). They share PGID with Spot train —
  # do NOT kill -9 until after Spot FPO++ python has exited.
  for pid in $pids; do
    kill -STOP "$pid" 2>/dev/null || true
    log "SIGSTOP 3tier shell pid=$pid"
  done
}

g1_teacher() {
  local ext
  ext="$(latest_ckpt /dev/shm/unitree_g1_ppo_gpu1/logs/rsl_rl/unitree_g1_29dof_velocity/*/model_*.pt)"
  if [[ -n "$ext" ]]; then
    echo "$ext"
  else
    echo "$G1_TEACHER_OLD"
  fi
}

# ---------- queues ----------
queue_gpu3_spot() {
  set -eo pipefail
  log "GPU3 queue: wait Spot FPO++ (live) → kd_only scratch → ours"
  # Wait until spot baseline train process is gone
  while pgrep -af 'fpo/train.py' | grep -q 'Isaac-Velocity-Flat-Spot-v0.*baseline'; do
    sleep 60
  done
  log "Spot FPO++ train exited"
  # Reap frozen 3tier waiters so they cannot spawn Ours (train already gone)
  pkill -CONT -f 'launch_3tier_resume.sh' 2>/dev/null || true
  sleep 1
  pkill -9 -f 'launch_3tier_resume.sh' 2>/dev/null || true
  sleep 5
  wait_gpu_idle 3 "Spot kd_only"

  GPU=3 FPO_VARIANT=kd_only EXPERIMENT_NAME=spot_kd_only \
    TEACHER_CHECKPOINT="$SPOT_TEACHER" \
    RUN_NAME="spot_kd_only_${STAMP}" WORKDIR=/dev/shm/spot_fpo_kd_only_gpu3 \
    NUM_ENVS=16384 MAX_ITERS=1500 \
    bash "${ROOT}/scripts/run_spot_fpo.sh" > "${PIPE}/gpu3_spot_kd_only.log" 2>&1
  log "Spot kd_only done; starting Ours"
  echo "SPOT_KD_ONLY_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"

  GPU=3 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=spot_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$SPOT_TEACHER" \
    RUN_NAME="spot_all_ideas_${STAMP}" WORKDIR=/dev/shm/spot_fpo_ours_gpu3 \
    NUM_ENVS=16384 MAX_ITERS=1500 \
    bash "${ROOT}/scripts/run_spot_fpo.sh" > "${PIPE}/gpu3_spot_ours.log" 2>&1
  log "OK GPU3 Spot kd_only + ours"
  echo "SPOT_OURS_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
  bash "${ROOT}/scripts/update_experiment_log_event.sh" "Spot FPO+++teacher(kd_only)+Ours 训练完成（待 play 填主表）" || true
}

queue_gpu1_g1() {
  set -eo pipefail
  log "GPU1 queue: wait G1 PPO extend → FPO++ → kd_only scratch → ours"
  while pgrep -af 'rsl_rl/train.py' | grep -q 'Unitree-G1-29dof-Velocity'; do
    sleep 60
  done
  log "G1 PPO extend exited"
  wait_gpu_idle 1 "G1 FPO++"

  local teacher fpo_ckpt
  teacher="$(g1_teacher)"
  fpo_ckpt="$(latest_ckpt /dev/shm/unitree_g1_fpo_baseline_gpu0/logs/fpo/unitree_g1_29dof_fpo_baseline/*/model_*.pt)"
  if [[ -n "$fpo_ckpt" ]]; then
    log "G1 FPO++ resume from $fpo_ckpt"
    # Remaining to 2000 (cfg default); if already >=1500, still ensure at least 500 more or skip if >=1990
    local it
    it="$(basename "$fpo_ckpt" .pt | sed 's/model_//')"
    if [[ "${it:-0}" -lt 1500 ]]; then
      GPU=1 FPO_VARIANT=baseline EXPERIMENT_NAME=unitree_g1_29dof_fpo_baseline \
        RESUME=1 CHECKPOINT="$fpo_ckpt" MAX_ITERS=$((1500 - it)) \
        RUN_NAME="g1_fpo_baseline_resume${it}_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_baseline_gpu1 \
        bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu1_g1_fpo_baseline.log" 2>&1
    else
      log "G1 FPO++ already at iter ${it} (≥1500); skip resume"
    fi
  else
    log "G1 FPO++ from scratch 1500"
    GPU=1 FPO_VARIANT=baseline EXPERIMENT_NAME=unitree_g1_29dof_fpo_baseline \
      MAX_ITERS=1500 RUN_NAME="g1_fpo_baseline_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_baseline_gpu1 \
      bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu1_g1_fpo_baseline.log" 2>&1
  fi
  echo "G1_FPO_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"

  wait_gpu_idle 1 "G1 kd_only"
  GPU=1 FPO_VARIANT=kd_only EXPERIMENT_NAME=unitree_g1_kd_only \
    TEACHER_CHECKPOINT="$teacher" MAX_ITERS=1500 \
    RUN_NAME="g1_kd_only_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_kd_only_gpu1 \
    bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu1_g1_kd_only.log" 2>&1
  echo "G1_KD_ONLY_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"

  wait_gpu_idle 1 "G1 ours"
  GPU=1 FPO_VARIANT=all_ideas_teacher_kd EXPERIMENT_NAME=unitree_g1_all_ideas_ppo_teacher \
    TEACHER_CHECKPOINT="$teacher" MAX_ITERS=1500 \
    RUN_NAME="g1_all_ideas_${STAMP}" WORKDIR=/dev/shm/unitree_g1_fpo_ours_gpu1 \
    bash "${ROOT}/scripts/run_g1_fpo.sh" > "${PIPE}/gpu1_g1_ours.log" 2>&1
  log "OK GPU1 G1 FPO++ + kd_only + ours"
  echo "G1_OURS_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
  bash "${ROOT}/scripts/update_experiment_log_event.sh" "G1 FPO+++kd_only+Ours 训练完成（待 play 填主表）" || true
}

queue_gpu0_h1_kd() {
  set -eo pipefail
  log "GPU0 queue: wait H1 Ours → H1 kd_only scratch (missing tier)"
  while pgrep -af 'fpo/train.py' | grep -q 'Unitree-H1-Velocity.*all_ideas_teacher_kd'; do
    sleep 60
  done
  log "H1 Ours exited"
  wait_gpu_idle 0 "H1 kd_only"

  GPU=0 FPO_VARIANT=kd_only EXPERIMENT_NAME=unitree_h1_kd_only \
    TEACHER_CHECKPOINT="$H1_TEACHER" MAX_ITERS=2000 \
    RUN_NAME="h1_kd_only_${STAMP}" WORKDIR=/dev/shm/unitree_h1_fpo_kd_only_gpu0 \
    bash "${ROOT}/scripts/run_h1_fpo.sh" > "${PIPE}/gpu0_h1_kd_only.log" 2>&1
  log "OK GPU0 H1 kd_only"
  echo "H1_KD_ONLY_DONE $(date -Iseconds)" >> "${PIPE}/KEEP_BUSY_EVENTS.txt"
  bash "${ROOT}/scripts/update_experiment_log_event.sh" "H1 FPO+++teacher(kd_only) 训练完成（待 play 填主表）" || true
}

# GPU2: when manip frees, run Go2 play fill / step sweeps that still need updating
queue_gpu2_evals() {
  set -eo pipefail
  log "GPU2 queue: wait manip free → fill Go2 kd_only note already in log; run Spot/H1 play when ckpts ready"
  while pgrep -af 'finetune_online_rl.py' | grep -q .; do
    sleep 120
  done
  wait_gpu_idle 2 "evals"

  # Prefer filling play tables for finished teachers when available
  if [[ -f "${ROOT}/scripts/eval_locomotion_play_fill.sh" ]]; then
    GPU=2 bash "${ROOT}/scripts/eval_locomotion_play_fill.sh" > "${PIPE}/gpu2_play_fill.log" 2>&1 || true
  else
    log "no eval_locomotion_play_fill.sh yet; idle-poll for next finished ckpt evals"
    # Lightweight: re-run Go2 kd_only step sweep already done — skip
    while true; do
      if gpu_idle 2; then
        # If Spot kd_only finished and no play yet, could add later
        sleep 300
      else
        sleep 120
      fi
      # exit when other queues done
      if [[ -f "${PIPE}/KEEP_BUSY_ALL_DONE" ]]; then
        break
      fi
    done
  fi
  log "OK GPU2 eval queue"
}

log "keep_busy start stamp=${STAMP}"
: > "${PIPE}/KEEP_BUSY_EVENTS.txt"
freeze_old_spot_waiter

queue_gpu3_spot &
echo $! > "${PIPE}/keep_busy_gpu3.pid"
queue_gpu1_g1 &
echo $! > "${PIPE}/keep_busy_gpu1.pid"
queue_gpu0_h1_kd &
echo $! > "${PIPE}/keep_busy_gpu0.pid"
queue_gpu2_evals &
echo $! > "${PIPE}/keep_busy_gpu2.pid"

wait
touch "${PIPE}/KEEP_BUSY_ALL_DONE"
log "keep_busy all queues finished"
}"
: > "${PIPE}/KEEP_BUSY_EVENTS.txt"
freeze_old_spot_waiter

queue_gpu3_spot &
echo $! > "${PIPE}/keep_busy_gpu3.pid"
queue_gpu1_g1 &
echo $! > "${PIPE}/keep_busy_gpu1.pid"
queue_gpu0_h1_kd &
echo $! > "${PIPE}/keep_busy_gpu0.pid"
queue_gpu2_evals &
echo $! > "${PIPE}/keep_busy_gpu2.pid"

wait
touch "${PIPE}/KEEP_BUSY_ALL_DONE"
log "keep_busy all queues finished"
