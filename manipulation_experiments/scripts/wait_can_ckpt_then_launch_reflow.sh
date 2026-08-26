#!/usr/bin/env bash
# Wait for a Can BC checkpoint, then launch 5 reflow-idea FPO++ finetunes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs/can_reflow_ideas"
LOG="$ROOT/logs/can_reflow_ideas/wait_launch_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$LOG") 2>&1

find_ckpt() {
  find "$ROOT/runs" -type d -path '*/flow_bc_can*/checkpoints/step_*' 2>/dev/null \
    | while read -r d; do
        if [[ -d "$d/policy" ]]; then
          echo "$d"
        fi
      done \
    | sort -V \
    | tail -1
}

echo "[$(date '+%F %T')] Waiting for Can BC checkpoint (runs/flow_bc_can*/checkpoints/step_*/policy)"
while true; do
  CKPT="$(find_ckpt || true)"
  if [[ -n "${CKPT}" ]]; then
    echo "[$(date '+%F %T')] Found ckpt: ${CKPT}"
    export CKPT_DIR="${CKPT}"
    export NUM_ENVS="${NUM_ENVS:-8}"
    export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
    export WANDB="${WANDB:-0}"
    bash "$ROOT/scripts/launch_can_reflow_ideas.sh"
    echo "[$(date '+%F %T')] Launch script returned."
    exit 0
  fi
  if ! pgrep -f 'pretrain_flow_bc.py.*flow_bc_can|pretrain_flow_bc.py.*Can' >/dev/null 2>&1; then
    CKPT="$(find_ckpt || true)"
    if [[ -n "${CKPT}" ]]; then
      export CKPT_DIR="${CKPT}"
      bash "$ROOT/scripts/launch_can_reflow_ideas.sh"
      exit 0
    fi
    echo "[$(date '+%F %T')] Can BC process gone and no checkpoint found. Exiting."
    exit 1
  fi
  sleep 60
done
