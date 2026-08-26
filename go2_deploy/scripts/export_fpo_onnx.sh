#!/usr/bin/env bash
# Export FPO Go2 checkpoint → ONNX (includes obs normalizer + flow act_inference).
# Usage:
#   bash scripts/export_fpo_onnx.sh --model baseline
#   MODEL=reflow bash scripts/export_fpo_onnx.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FGO="$(cd "${ROOT}/.." && pwd)"
ISAAC="${FGO}/isaaclab_experiments"
MODEL_NAME="${MODEL:-baseline}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_NAME="$2"; shift 2 ;;
    --steps) SAMPLING_STEPS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PY="${ISAAC}/thirdparty/miniconda3/envs/isaaclab_fpo/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "isaaclab_fpo python not found at ${PY}" >&2
  exit 1
fi

CKPT="$(
  cd "${ROOT}"
  "${ROOT}/.venv/bin/python" - <<PY
from go2_deploy.config import load_deploy_config
cfg = load_deploy_config(model_override="${MODEL_NAME}")
m = cfg.model
print(m.checkpoint)
assert m.checkpoint.is_file(), m.checkpoint
PY
)"

OUT_DIR="${ROOT}/exported/${MODEL_NAME}"
mkdir -p "${OUT_DIR}"
echo "[export] model=${MODEL_NAME} steps=${SAMPLING_STEPS}"
echo "[export] ckpt=${CKPT}"
echo "[export] out=${OUT_DIR}/policy.onnx"

export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${ISAAC}/isaaclab_fpo:${PYTHONPATH:-}"

"${PY}" - <<PY
import os, sys
from pathlib import Path
import torch
torch._dynamo.config.disable = True

sys.path.insert(0, "${ISAAC}/isaaclab_fpo")
from isaaclab_fpo.modules.actor_critic import ActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg
from isaaclab_fpo.exporter import export_policy_as_onnx
from rsl_rl.modules import EmpiricalNormalization

ckpt = Path("${CKPT}")
d = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg = FpoRslRlPpoActorCriticCfg(
    class_name="ActorCritic",
    init_noise_std=1.0,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[768, 768, 768],
    activation="elu",
    actor_scale=1.0,
    actor_mlp_output_scale=1.0,
    timestep_embed_dim=8,
    sampling_steps=int("${SAMPLING_STEPS}"),
    cfm_loss_reduction="sqrt",
    action_perturb_std=0.02,
)
policy = ActorCritic(num_actor_obs=48, num_critic_obs=48, num_actions=12, cfg=cfg)
policy.load_state_dict(d["model_state_dict"], strict=True)
policy._compiled_integrate_flow = policy._integrate_flow
norm = EmpiricalNormalization(shape=[48], until=1.0e8)
norm.load_state_dict(d["obs_norm_state_dict"])
policy.eval()
norm.eval()
out = Path("${OUT_DIR}")
out.mkdir(parents=True, exist_ok=True)
export_policy_as_onnx(policy, path=str(out), normalizer=norm, filename="policy.onnx")
print("[export] wrote", out / "policy.onnx", "bytes", (out / "policy.onnx").stat().st_size)
with torch.no_grad():
    a = policy.act_inference(norm(torch.zeros(1, 48)))
print("[export] smoke action[:4]=", a[0, :4].tolist())
PY

echo "[export] Update configs/deploy.yaml: policy.onnx_path: exported/${MODEL_NAME}/policy.onnx"
echo "[export] Run: python -m go2_deploy --mode fpo --model ${MODEL_NAME}"
