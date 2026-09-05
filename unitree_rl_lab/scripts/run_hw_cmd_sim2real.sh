#!/usr/bin/env bash
# Replay 3 hardware cmd trajectories in Isaac and compute sim↔real similarity.
# HW pack used ours_s1 → default sampling_steps=1.
#
# Usage:
#   bash scripts/run_hw_cmd_sim2real.sh
#   GPU=2 STEPS=1 bash scripts/run_hw_cmd_sim2real.sh
#   bash scripts/run_hw_cmd_sim2real.sh --viser-run 1   # after metrics: open Viser for run1
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

PACK="${HW_PACK:-/workspace/fgo_test/cmd_traj_runs_all}"
CKPT="${CHECKPOINT:-${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300/model_1499.pt}"
STEPS="${STEPS:-1}"
GPU="${GPU:-2}"
PORT="${PORT:-8081}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/fpo/hw_sim2real_$(date +%Y%m%d_%H%M%S)}"

# Latest canonical trio from cmd_traj_runs_all (fwd/back, left/right, yaw).
RUNS=(
  "run4:${PACK}/cmd_traj_run4_20260904_155100.npz"
  "run5:${PACK}/cmd_traj_run5_20260904_155220.npz"
  "run6:${PACK}/cmd_traj_run6_20260904_155333.npz"
)

VISER_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --viser-run) VISER_RUN="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --all6)
      RUNS=(
        "run1:${PACK}/cmd_traj_run1_20260904_152714.npz"
        "run2:${PACK}/cmd_traj_run2_20260904_152956.npz"
        "run3:${PACK}/cmd_traj_run3_20260904_153112.npz"
        "run4:${PACK}/cmd_traj_run4_20260904_155100.npz"
        "run5:${PACK}/cmd_traj_run5_20260904_155220.npz"
        "run6:${PACK}/cmd_traj_run6_20260904_155333.npz"
      )
      shift
      ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUT_ROOT}"
echo "[$(date '+%F %T')] out=${OUT_ROOT} steps=${STEPS} gpu=${GPU} pack=${PACK}"

if [[ -n "${VISER_RUN}" ]]; then
  case "${VISER_RUN}" in
    1) NPZ="${PACK}/cmd_traj_run1_20260904_152714.npz" ;;
    2) NPZ="${PACK}/cmd_traj_run2_20260904_152956.npz" ;;
    3) NPZ="${PACK}/cmd_traj_run3_20260904_153112.npz" ;;
    4) NPZ="${PACK}/cmd_traj_run4_20260904_155100.npz" ;;
    5) NPZ="${PACK}/cmd_traj_run5_20260904_155220.npz" ;;
    6) NPZ="${PACK}/cmd_traj_run6_20260904_155333.npz" ;;
    *) echo "VISER_RUN must be 1..6"; exit 1 ;;
  esac
  echo "[$(date '+%F %T')] Viser HW cmd replay run${VISER_RUN} → http://localhost:${PORT}"
  echo "  ssh -L ${PORT}:localhost:${PORT} <host>"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u "${ROOT}/scripts/fpo/play_with_viser.py" \
    --task Unitree-Go2-Velocity \
    --headless \
    --device cuda:0 \
    --num_envs 1 \
    --fpo_variant all_ideas_teacher_kd \
    --checkpoint "${CKPT}" \
    --flow-sampling-steps "${STEPS}" \
    --viser \
    --viser-port "${PORT}" \
    --asset-dir "${ROOT}/viser_assets/unitree_go2_velocity" \
    --cmd-npz "${NPZ}" \
    --real-time
  exit 0
fi

for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  npz="${entry#*:}"
  out="${OUT_ROOT}/${name}"
  mkdir -p "${out}"
  echo "[$(date '+%F %T')] === ${name} === ${npz}"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u "${ROOT}/scripts/fpo/replay_hw_cmd_traj.py" \
    --task Unitree-Go2-Velocity \
    --headless \
    --device cuda:0 \
    --num_envs 1 \
    --fpo_variant all_ideas_teacher_kd \
    --checkpoint "${CKPT}" \
    --sampling-steps "${STEPS}" \
    --hw-npz "${npz}" \
    --out_dir "${out}"
done

python - <<PY
import json, pathlib
root = pathlib.Path("${OUT_ROOT}")
rows = []
for p in sorted(root.glob("run*/similarity.json")):
    name = p.parent.name
    m = json.loads(p.read_text())
    rows.append({
        "run": name,
        "joint_pos_corr": m["joint_pos"]["corr"],
        "joint_pos_mae": m["joint_pos"]["mae"],
        "joint_vel_corr": m["joint_vel"]["corr"],
        "joint_vel_mae": m["joint_vel"]["mae"],
        "gyro_corr": m["gyro"]["corr"],
        "gyro_mae": m["gyro"]["mae"],
        "sim_q_vs_hw_q_cmd_corr": m["sim_q_vs_hw_q_cmd"]["corr"],
        "path": str(p),
    })
    print(f"{name}: joint_corr={m['joint_pos']['corr']:.4f} mae={m['joint_pos']['mae']:.4f} "
          f"gyro_corr={m['gyro']['corr']:.4f} mae={m['gyro']['mae']:.4f}")
out = {"sampling_steps": int("${STEPS}"), "pack": "${PACK}", "runs": rows}
(root / "SUMMARY.json").write_text(json.dumps(out, indent=2))
print("SUMMARY →", root / "SUMMARY.json")
PY

echo
echo "View sim (Viser) with the SAME HW cmds:"
echo "  bash scripts/run_hw_cmd_sim2real.sh --viser-run 4   # fwd/back (new)"
echo "  bash scripts/run_hw_cmd_sim2real.sh --viser-run 5   # left/right (new)"
echo "  bash scripts/run_hw_cmd_sim2real.sh --viser-run 6   # yaw (new)"
echo "  then open http://localhost:${PORT} (ssh -L ${PORT}:localhost:${PORT} ...)"
