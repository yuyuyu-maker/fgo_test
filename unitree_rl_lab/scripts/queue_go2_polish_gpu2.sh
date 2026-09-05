#!/usr/bin/env bash
# GPU2 Go2 Ours polish: resume model_1499 → kd_coef=0.2, zero_x0=0.4, iters→2500 → play gate.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE="${ROOT}/logs/fpo/gpu_queues"
mkdir -p "$PIPE"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG="${PIPE}/queue_go2_polish_gpu2_${STAMP}.out"
exec > >(tee -a "$LOG") 2>&1

GPU=2
OURS_CKPT="${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300/model_1499.pt"
PPO_TEACHER="${ROOT}/logs/rsl_rl/unitree_go2_velocity/2026-09-02_17-20-40_unitree_go2_ppo/model_2999.pt"
WORKDIR="/dev/shm/unitree_go2_ours_polish_gpu2"
PLAY_OUT="${PIPE}/go2_polish_play_${STAMP}"
# Old main-table baselines for replace gate
OLD_ZERO64=34.78
OLD_ZERO1=34.25

log() { echo "[$(date '+%F %T')] $*"; }

echo "$$" > "${PIPE}/queue_go2_polish_gpu2.pid"
log "Go2 polish GPU2 stamp=${STAMP}"
[[ -f "${OURS_CKPT}" ]] || { log "ABORT missing Ours ckpt ${OURS_CKPT}"; exit 1; }
[[ -f "${PPO_TEACHER}" ]] || { log "ABORT missing PPO teacher ${PPO_TEACHER}"; exit 1; }

log "train polish resume → +1001 iters (~absolute 2500) kd=0.2 zero_x0=0.4"
# NOTE: runner does tot_iter = start_iter + max_iterations (additive on resume).
GPU=2 FPO_VARIANT=all_ideas_teacher_kd NUM_ENVS=4096 MAX_ITERS=1001 \
  RESUME=1 CHECKPOINT="${OURS_CKPT}" \
  TEACHER_CHECKPOINT="${PPO_TEACHER}" \
  TEACHER_KD_COEF=0.2 \
  TEACHER_AUX_ZERO_X0_PROB=0.4 \
  EXPERIMENT_NAME=unitree_go2_all_ideas_ppo_teacher \
  RUN_NAME="go2_ours_polish_kd02_z04_2500_${STAMP}" \
  WORKDIR="${WORKDIR}" \
  bash "${ROOT}/scripts/run_go2_fpo.sh" \
  > "${PIPE}/gpu2_go2_polish_${STAMP}.log" 2>&1
train_rc=$?
log "train exit=${train_rc}"

POLISH_CKPT="$(ls -1t ${WORKDIR}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/*/model_*.pt 2>/dev/null | head -1 || true)"
if [[ -z "${POLISH_CKPT}" ]]; then
  POLISH_CKPT="$(ls -1t ${ROOT}/logs/fpo/unitree_go2_all_ideas_ppo_teacher/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
log "polish ckpt=${POLISH_CKPT:-MISSING}"
[[ -n "${POLISH_CKPT}" ]] || { log "ABORT no polish ckpt"; exit 1; }

mkdir -p "${PLAY_OUT}"
log "play zero/random @64 1 → ${PLAY_OUT}"
set +u
source /workspace/fgo_test/isaaclab_experiments/source_env.sh
set -uo pipefail
export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
unset CUDA_VISIBLE_DEVICES || true

python -u "${ROOT}/scripts/fpo/eval_sampling_steps.py" \
  --task Unitree-Go2-Velocity --headless --device "cuda:${GPU}" \
  --num_envs 256 --eval_episodes 10 --sampling_steps 64 1 --eval_modes zero random \
  --fpo_variant all_ideas_teacher_kd \
  --model "ours_polish=${POLISH_CKPT}" \
  > "${PLAY_OUT}/ours_polish.log" 2>&1 || log "FAIL play"

# Parse means from log / summary if present
SUMMARY="$(ls -1t ${PLAY_OUT}/*summary* ${ROOT}/logs/fpo/*polish*/*summary* 2>/dev/null | head -1 || true)"
log "play log=${PLAY_OUT}/ours_polish.log summary=${SUMMARY:-none}"
rg -n 'zero|random|mean|64|1-step|ours_polish' "${PLAY_OUT}/ours_polish.log" | tail -80 || true

# Gate note (docs update done by agent after numbers parsed)
export PLAY_OUT POLISH_CKPT OLD_ZERO64 OLD_ZERO1
python3 - <<'PY'
import os, re, pathlib
play_out = os.environ["PLAY_OUT"]
ckpt = os.environ.get("POLISH_CKPT", "")
old_z64 = float(os.environ["OLD_ZERO64"])
logp = pathlib.Path(play_out) / "ours_polish.log"
text = logp.read_text(errors="replace") if logp.exists() else ""
nums = {}
for mode in ("zero", "random"):
    for steps in (64, 1):
        pats = [
            rf"ours_polish/{mode}\s*@?\s*{steps}\s*[:=]\s*(-?\d+\.?\d*)",
            rf"{mode}\s*@\s*{steps}\s*[:=]\s*(-?\d+\.?\d*)",
            rf"steps?\s*=\s*{steps}.*?{mode}.*?(-?\d+\.\d+)",
        ]
        for p in pats:
            m = re.search(p, text, re.I | re.S)
            if m:
                nums[f"{mode}@{steps}"] = float(m.group(1))
                break
print("PARSED", nums)
z64, z1 = nums.get("zero@64"), nums.get("zero@1")
gate = pathlib.Path(play_out) / "REPLACE_GATE.txt"
if z64 is None or z1 is None:
    gate.write_text(f"INDETERMINATE parse_failed nums={nums}\nckpt={ckpt}\n")
    print("GATE INDETERMINATE")
elif z64 >= old_z64 and z1 >= 34.0:
    gate.write_text(
        f"REPLACE_OK zero@64={z64} (>= {old_z64}) zero@1={z1} (>=34.0)\n"
        f"ckpt={ckpt}\nnums={nums}\n"
    )
    print("GATE REPLACE_OK", z64, z1)
else:
    gate.write_text(
        f"KEEP_OLD zero@64={z64} (need>={old_z64}) zero@1={z1} (need>=34.0)\n"
        f"ckpt={ckpt}\nnums={nums}\n"
    )
    print("GATE KEEP_OLD", z64, z1)
PY

log "DONE stamp=${STAMP} log=${LOG}"
