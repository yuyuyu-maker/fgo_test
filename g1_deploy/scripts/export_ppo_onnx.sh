#!/usr/bin/env bash
# Export G1 Velocity PPO checkpoint → ONNX into this package (velocity/v1_ours).
#
# Usage:
#   bash scripts/export_ppo_onnx.sh /path/to/model_9999.pt
#   CHECKPOINT=/path/to/model_9999.pt bash scripts/export_ppo_onnx.sh
#   # or auto-pick latest under a run dir:
#   bash scripts/export_ppo_onnx.sh --run-dir /dev/shm/unitree_g1_ppo_gpu1/...
#
# Needs Isaac Lab env: source ../isaaclab_experiments/source_env.sh (or set PYTHON)
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FGO="$(cd "${ROOT}/.." && pwd)"
RL_LAB="${UNITREE_RL_LAB:-${FGO}/unitree_rl_lab}"
TASK="${TASK:-Unitree-G1-29dof-Velocity}"
SLOT="${POLICY_SLOT:-v1_ours}"
OUT_SLOT="${ROOT}/robots/g1_29dof/config/policy/velocity/${SLOT}"
GPU="${GPU:-0}"

CKPT="${CHECKPOINT:-}"
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --checkpoint|--ckpt) CKPT="$2"; shift 2 ;;
    --slot) SLOT="$2"; OUT_SLOT="${ROOT}/robots/g1_29dof/config/policy/velocity/${SLOT}"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    -*)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
    *)
      CKPT="$1"
      shift
      ;;
  esac
done

if [[ -z "${CKPT}" && -n "${RUN_DIR}" ]]; then
  CKPT="$(ls -1 "${RUN_DIR}"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Usage: $0 /path/to/model_XXXX.pt" >&2
  echo "   or: $0 --run-dir /path/to/logdir" >&2
  exit 1
fi
CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")"

# Prefer project isaaclab python if available
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${FGO}/isaaclab_experiments/thirdparty/miniconda3/envs/isaaclab/bin/python" ]]; then
    PYTHON="${FGO}/isaaclab_experiments/thirdparty/miniconda3/envs/isaaclab/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    PYTHON="$(command -v python3)"
  fi
fi

echo "[export] ckpt=${CKPT}"
echo "[export] task=${TASK}"
echo "[export] python=${PYTHON}"
echo "[export] slot=${OUT_SLOT}"

mkdir -p "${OUT_SLOT}/params" "${OUT_SLOT}/exported" "${ROOT}/exported/${SLOT}"

# CRITICAL: use training-exported deploy.yaml (obs scales + kp/kd), NOT official v0.
# Official v0 differs in waist stiffness (200 vs 40) and arm damping (10 vs 1).
TRAIN_DEPLOY="$(dirname "${CKPT}")/params/deploy.yaml"
if [[ -f "${TRAIN_DEPLOY}" ]]; then
  cp -f "${TRAIN_DEPLOY}" "${OUT_SLOT}/params/deploy.yaml"
  echo "[export] Using training deploy.yaml: ${TRAIN_DEPLOY}"
else
  echo "[export] WARN: ${TRAIN_DEPLOY} missing; falling back to official v0 (kp/kd may mismatch training)"
  cp -f "${ROOT}/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml" \
     "${OUT_SLOT}/params/deploy.yaml"
fi

cd "${RL_LAB}"
export UNITREE_EXPORT_ONLY=1
export CUDA_VISIBLE_DEVICES="${GPU}"

# play.py exports next to the checkpoint as <ckpt_dir>/exported/policy.onnx
"${PYTHON}" scripts/rsl_rl/play.py \
  --task "${TASK}" \
  --checkpoint "${CKPT}" \
  --num_envs 1 \
  --headless

EXPORTED_SRC="$(dirname "${CKPT}")/exported"
if [[ ! -f "${EXPORTED_SRC}/policy.onnx" ]]; then
  echo "[export] ERROR: expected ${EXPORTED_SRC}/policy.onnx" >&2
  exit 1
fi

cp -f "${EXPORTED_SRC}/policy.onnx" "${OUT_SLOT}/exported/policy.onnx"
cp -f "${EXPORTED_SRC}/policy.onnx" "${ROOT}/exported/${SLOT}/policy.onnx"
# Re-copy train deploy.yaml after export (do not overwrite with official)
if [[ -f "${TRAIN_DEPLOY}" ]]; then
  cp -f "${TRAIN_DEPLOY}" "${OUT_SLOT}/params/deploy.yaml"
  cp -f "${TRAIN_DEPLOY}" "${ROOT}/exported/${SLOT}/deploy.yaml"
fi

echo "[export] Installed -> ${OUT_SLOT}/exported/policy.onnx"
echo "[export] deploy.yaml -> ${OUT_SLOT}/params/deploy.yaml (from training)"
echo "[export] parser_policy_dir will prefer '${SLOT}' over 'v0' (lexicographic last with exported/)."
echo "[export] To force official: set Velocity.policy_dir to config/policy/velocity/v0 in config.yaml"
echo "[export] Rebuild not required; restart g1_ctrl to load new ONNX."
