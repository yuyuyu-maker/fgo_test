#!/usr/bin/env python3
"""Export Go2 Ours (all_ideas_teacher_kd) to one ONNX per sampling-step budget.

Bakes Euler flow steps into the graph (obs[45] -> actions[12]).
Usage:
  python scripts/fpo/export_go2_ours_steps.py
  python scripts/fpo/export_go2_ours_steps.py --steps 64 32 1 --checkpoint /path/model_1499.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))

import torch

torch._dynamo.config.disable = True

from isaaclab_fpo.exporter import export_policy_as_onnx
from isaaclab_fpo.modules.actor_critic import ActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg

DEFAULT_CKPT = (
    _ROOT
    / "logs/fpo/unitree_go2_all_ideas_ppo_teacher"
    / "2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300"
    / "model_1499.pt"
)
DEFAULT_STEPS = (64, 32, 16, 8, 4, 1)


def export_one(ckpt: Path, steps: int, out_dir: Path, *, obs_dim: int, critic_dim: int, act_dim: int) -> Path:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = FpoRslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        actor_scale=1.0,
        actor_mlp_output_scale=1.0,
        timestep_embed_dim=8,
        sampling_steps=int(steps),
        cfm_loss_reduction="sqrt",
        action_perturb_std=0.02,
        # Match Ours weights (step_predictor present); inference ONNX uses fixed ``steps``.
        adaptive_compute_enabled=True,
    )
    policy = ActorCritic(obs_dim, critic_dim, act_dim, cfg)
    missing, unexpected = policy.load_state_dict(blob["model_state_dict"], strict=False)
    if missing:
        print(f"[WARN] s={steps} missing keys: {missing}")
    if unexpected:
        print(f"[WARN] s={steps} unexpected keys: {unexpected}")
    policy.sampling_steps = int(steps)
    policy._compiled_integrate_flow = policy._integrate_flow
    policy.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    # C++ unitree_rl_lab expects: <policy_dir>/exported/policy.onnx + params/deploy.yaml
    export_dir = out_dir / "exported"
    params_dir = out_dir / "params"
    export_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    export_policy_as_onnx(policy, path=str(export_dir), normalizer=None, filename="policy.onnx")
    onnx_path = export_dir / "policy.onnx"

    with torch.no_grad():
        a = policy.act_inference(torch.zeros(1, obs_dim))
    print(f"[OK] steps={steps} -> {onnx_path} ({onnx_path.stat().st_size} bytes) smoke={a[0, :4].tolist()}")

    meta = out_dir / "export_meta.txt"
    meta.write_text(
        f"checkpoint={ckpt}\n"
        f"sampling_steps={steps}\n"
        f"obs_dim={obs_dim}\n"
        f"action_dim={act_dim}\n"
        f"variant=all_ideas_teacher_kd\n"
        f"obs_layout=unitree_rl_lab Go2 (45-D, NO base_lin_vel; see params/deploy.yaml)\n"
        f"note=flow Euler steps baked into ONNX; adaptive predictor weights loaded but unused in graph\n"
    )
    return onnx_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Root for ours_s{N}/policy.onnx (default: <ckpt_dir>/exported_steps + go2_deploy mirror)",
    )
    p.add_argument("--obs-dim", type=int, default=45)
    p.add_argument("--critic-obs-dim", type=int, default=60)
    p.add_argument("--action-dim", type=int, default=12)
    p.add_argument("--skip-go2-deploy-mirror", action="store_true")
    args = p.parse_args()

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    out_root = args.out_root.resolve() if args.out_root else (ckpt.parent / "exported_steps")
    go2_deploy = _ROOT.parent / "go2_deploy" / "exported"

    print(f"[INFO] ckpt={ckpt}")
    print(f"[INFO] steps={args.steps}")
    print(f"[INFO] out_root={out_root}")

    for s in args.steps:
        dest = out_root / f"ours_s{s}"
        path = export_one(ckpt, s, dest, obs_dim=args.obs_dim, critic_dim=args.critic_obs_dim, act_dim=args.action_dim)
        dy = ckpt.parent / "params" / "deploy.yaml"
        if dy.is_file():
            (dest / "params" / "deploy.yaml").write_text(dy.read_text())
        if not args.skip_go2_deploy_mirror and go2_deploy.parent.is_dir():
            mirror = go2_deploy / f"ours_s{s}"
            mirror_exp = mirror / "exported"
            mirror_par = mirror / "params"
            mirror_exp.mkdir(parents=True, exist_ok=True)
            mirror_par.mkdir(parents=True, exist_ok=True)
            (mirror_exp / "policy.onnx").write_bytes(path.read_bytes())
            # Flat copy for go2_deploy auto path exported/<model>/policy.onnx
            (mirror / "policy.onnx").write_bytes(path.read_bytes())
            (mirror / "export_meta.txt").write_text((dest / "export_meta.txt").read_text())
            if dy.is_file():
                (mirror_par / "deploy.yaml").write_text(dy.read_text())
                (mirror / "deploy.yaml").write_text(dy.read_text())
            print(f"[OK] mirrored -> {mirror_exp / 'policy.onnx'}")

    print("[DONE] all step ONNX exports")


if __name__ == "__main__":
    main()
