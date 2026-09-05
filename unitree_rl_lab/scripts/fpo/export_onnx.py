#!/usr/bin/env python3
"""Export FPO++ ActorCritic checkpoint to ONNX for official Unitree C++ deploy.

No Isaac Sim. Graph is obs(45) -> actions(12) with Euler flow baked in (input name "obs").
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--obs-dim", type=int, default=45)
    parser.add_argument("--critic-obs-dim", type=int, default=60)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--sampling-steps", type=int, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ckpt.parent / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        sampling_steps=int(args.sampling_steps or 64),
        cfm_loss_reduction="sqrt",
        action_perturb_std=0.02,
        adaptive_compute_enabled=False,
    )
    policy = ActorCritic(args.obs_dim, args.critic_obs_dim, args.action_dim, cfg)
    missing, unexpected = policy.load_state_dict(blob["model_state_dict"], strict=False)
    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)
    policy._compiled_integrate_flow = policy._integrate_flow
    policy.eval()

    export_policy_as_onnx(policy, path=str(out_dir), normalizer=None, filename="policy.onnx")
    onnx_path = out_dir / "policy.onnx"
    print(f"[INFO] wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")

    with torch.no_grad():
        a = policy._integrate_flow(
            torch.zeros(1, args.obs_dim),
            torch.zeros(1, args.action_dim),
            torch.linspace(1.0, 0.0, cfg.sampling_steps + 1)[:-1],
            torch.linspace(1.0, 0.0, cfg.sampling_steps + 1)[1:]
            - torch.linspace(1.0, 0.0, cfg.sampling_steps + 1)[:-1],
            cfg.sampling_steps,
        )
    print("[INFO] smoke action[:4]=", a[0, :4].tolist())


if __name__ == "__main__":
    main()
