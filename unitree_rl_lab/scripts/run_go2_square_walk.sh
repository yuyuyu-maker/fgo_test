#!/usr/bin/env bash
# Run Go2 Ours square-velocity walk in Isaac Lab (0.3 m/s × 3s per side).
# Usage:
#   bash scripts/run_go2_square_walk.sh
#   bash scripts/run_go2_square_walk.sh --steps 8 --cycles 2
#   STEPS="64 32 16 8 4 1" bash scripts/run_go2_square_walk.sh --all-steps
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export TORCHDYNAMO_DISABLE=1

CKPT="${CHECKPOINT:-${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300/model_1499.pt}"
SPEED="${SPEED:-0.3}"
SEGMENT_S="${SEGMENT_S:-3.0}"
CYCLES="${CYCLES:-3}"
GPU="${GPU:-0}"
ALL=0
ONE_STEPS=64
HEADLESS=1
VIDEO=0
# Isaac Lab livestream: 0=off, 1=public WebRTC, 2=private WebRTC (browser)
LIVESTREAM="${LIVESTREAM:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) ONE_STEPS="$2"; shift 2 ;;
    --all-steps) ALL=1; shift ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --speed) SPEED="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --gui|--viz) HEADLESS=0; LIVESTREAM=0; shift ;;
    --web)
      echo "ERROR: --web / Isaac WebRTC livestream 在本机不可用。" >&2
      echo "  原因: Vulkan 初始化失败 (ERROR_INCOMPATIBLE_DRIVER)，rendering.kit 会 segfault。" >&2
      echo "  请改用纯数据模式:" >&2
      echo "    bash scripts/run_go2_square_walk.sh --steps 64" >&2
      echo "  若要看画面: 在有正常显示/Vulkan 的机器上跑 --gui，或拷 rollout 图本地看。" >&2
      exit 2
      ;;
    --livestream)
      echo "ERROR: --livestream 同 --web，本机 Vulkan/WebRTC 不可用。请用无参 headless 跑。" >&2
      exit 2
      ;;
    --video) VIDEO=1; shift ;;
    --headless) HEADLESS=1; LIVESTREAM=0; shift ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

# Sanity: print what we will run
echo "[$(date '+%F %T')] plan steps=${ONE_STEPS} all=${ALL} livestream=${LIVESTREAM} headless=${HEADLESS} gpu=${GPU}"
echo "[$(date '+%F %T')] ckpt=${CKPT}"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

run_one() {
  local s="$1"
  local extra=()
  if [[ "${LIVESTREAM}" != "0" ]]; then
    extra+=(--livestream "${LIVESTREAM}" --enable_cameras)
    echo "[$(date '+%F %T')] WebRTC livestream=${LIVESTREAM} — open Isaac WebRTC / Streaming Client to this machine"
  elif [[ "${HEADLESS}" == "1" ]]; then
    extra+=(--headless)
  else
    extra+=(--real-time)
  fi
  if [[ "${VIDEO}" == "1" ]]; then
    extra+=(--video --enable_cameras)
  fi
  echo "[$(date '+%F %T')] square walk sampling_steps=${s} headless=${HEADLESS} livestream=${LIVESTREAM} video=${VIDEO}"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/scripts/fpo/play_go2_square_walk.py" \
    --task Unitree-Go2-Velocity \
    --fpo_variant all_ideas_teacher_kd \
    --checkpoint "${CKPT}" \
    --sampling-steps "${s}" \
    --speed "${SPEED}" \
    --segment_s "${SEGMENT_S}" \
    --cycles "${CYCLES}" \
    --num_envs 1 \
    --device cuda:0 \
    "${extra[@]}"
}

cd "${ROOT}"
if [[ "${ALL}" == "1" ]]; then
  for s in ${STEPS:-64 32 16 8 4 1}; do
    run_one "$s"
  done
else
  run_one "${ONE_STEPS}"
fi
