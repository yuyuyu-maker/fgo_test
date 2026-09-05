#!/usr/bin/env bash
# Play G1 PPO in Viser with official URDF meshes.
# Usage:
#   GPU=2 PORT=8082 bash scripts/play_g1_ppo_viser.sh
#   CHECKPOINT=.../model_9998.pt bash scripts/play_g1_ppo_viser.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${ROOT}/scripts/rsl_rl:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES || true

GPU="${GPU:-2}"
PORT="${PORT:-8082}"
NUM_ENVS="${NUM_ENVS:-1}"
CKPT="${CHECKPOINT:-${ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity/2026-09-04_11-15-17_g1_ppo_extend_curriculum_2026-09-04_11-14-23/model_9300.pt}"
ASSET_DIR="${ASSET_DIR:-${ROOT}/viser_assets/unitree_g1_29dof_velocity}"

cd "$ROOT"
echo "[$(date '+%F %T')] G1 PPO Viser port=${PORT} gpu=${GPU}"
echo "  ckpt=${CKPT}"
echo "  assets=${ASSET_DIR} (from unitree_ros g1_29dof_rev_1_0.urdf)"
echo "  open http://localhost:${PORT} (ssh -L ${PORT}:localhost:${PORT} ...)"

python -u "${ROOT}/scripts/rsl_rl/play_g1_ppo_viser.py" \
  --task Unitree-G1-29dof-Velocity \
  --headless \
  --device "cuda:${GPU}" \
  --num_envs "${NUM_ENVS}" \
  --checkpoint "${CKPT}" \
  --viser \
  --viser-port "${PORT}" \
  --asset-dir "${ASSET_DIR}" \
  --real-time
