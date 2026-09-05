#!/usr/bin/env bash
# Export the latest Unitree-Go2-Velocity PPO checkpoint to policy.onnx + deploy.yaml.
# Usage:
#   GPU=1 bash scripts/export_go2_onnx.sh
#   CHECKPOINT=/path/to/model_2999.pt GPU=1 bash scripts/export_go2_onnx.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /workspace/fgo_test/isaaclab_experiments/source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
unset CUDA_VISIBLE_DEVICES || true

GPU="${GPU:-1}"
TASK="Unitree-Go2-Velocity"
LOG_ROOT="${ROOT}/logs/rsl_rl/unitree_go2_velocity"
if [[ -d /dev/shm/unitree_go2_ppo/logs/rsl_rl/unitree_go2_velocity ]]; then
  LOG_ROOT="/dev/shm/unitree_go2_ppo/logs/rsl_rl/unitree_go2_velocity"
fi

ARGS=(--task "$TASK" --headless --num_envs 16 --device "cuda:${GPU}")
if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

cd /dev/shm/unitree_go2_ppo
# play.py looks for logs/rsl_rl/<experiment> relative to cwd
mkdir -p logs/rsl_rl
ln -sfn "$LOG_ROOT" logs/rsl_rl/unitree_go2_velocity

python "${ROOT}/scripts/rsl_rl/play.py" "${ARGS[@]}"

# Copy ONNX next to the checkpoint and to a stable folder for the real robot.
EXPORTED=$(find "$LOG_ROOT" -name policy.onnx -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
if [[ -z "$EXPORTED" ]]; then
  echo "policy.onnx not found under $LOG_ROOT"
  exit 1
fi
RUN_DIR="$(dirname "$(dirname "$EXPORTED")")"
DEST="${ROOT}/deploy/robots/go2/exported"
mkdir -p "$DEST"
cp -f "$EXPORTED" "$DEST/policy.onnx"
cp -f "$RUN_DIR/params/deploy.yaml" "$DEST/deploy.yaml" 2>/dev/null || true
echo "exported: $DEST/policy.onnx"
echo "deploy.yaml: $DEST/deploy.yaml"
echo "checkpoint dir: $RUN_DIR"
