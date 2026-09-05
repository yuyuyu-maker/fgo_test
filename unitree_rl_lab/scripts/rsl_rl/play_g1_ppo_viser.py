#!/usr/bin/env python3
"""Play Unitree G1 PPO checkpoint with Viser (proxy meshes OK if GLBs missing)."""
from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))
sys.path.insert(0, str(_ROOT / "scripts" / "rsl_rl"))

import viser  # noqa: F401

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play G1 PPO with Viser.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--viser", action="store_true", default=True)
parser.add_argument("--no-viser", action="store_false", dest="viser")
parser.add_argument("--viser-port", type=int, default=8082)
parser.add_argument("--asset-dir", type=str, default=None)
parser.add_argument("--viser-update-freq", type=int, default=1)
parser.add_argument("--viser-env-spacing", type=float, default=1.5)
parser.add_argument("--viser-fps", type=int, default=50)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo.viser import ViserIsaacLab


def _default_asset_dir() -> pathlib.Path:
    candidates = [
        _ROOT / "viser_assets" / "unitree_g1_29dof_velocity",
        pathlib.Path(
            "/workspace/fgo_test/isaaclab_experiments/isaaclab_fpo/viser_assets/isaac_velocity_flat_g1_v0"
        ),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped

    viser_viz = None
    if args_cli.viser:
        asset_dir = pathlib.Path(args_cli.asset_dir) if args_cli.asset_dir else _default_asset_dir()
        print(f"[INFO] Viser assets: {asset_dir} (GLB missing → proxy boxes OK)")
        try:
            viser_viz = ViserIsaacLab(
                asset_dir=asset_dir,
                port=int(args_cli.viser_port),
                update_freq=int(args_cli.viser_update_freq),
                num_envs=min(env_cfg.scene.num_envs, 16),
                env_spacing=float(args_cli.viser_env_spacing),
                fps=int(args_cli.viser_fps),
            )
            viser_viz.load_from_env(base_env)
            print(f"[INFO] Viser at http://0.0.0.0:{args_cli.viser_port}")
        except Exception as exc:
            print(f"[ERROR] Viser init failed: {exc}")
            import traceback

            traceback.print_exc()
            viser_viz = None

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading G1 PPO ckpt: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    got = env.get_observations()
    obs = got[0] if isinstance(got, tuple) else got

    while simulation_app.is_running():
        start = time.time()
        reset_requested = viser_viz is not None and viser_viz.check_reset_request()
        with torch.inference_mode():
            actions = policy(obs)
            step_out = env.step(actions)
            obs, rewards = step_out[0], step_out[1]
            if isinstance(obs, tuple):
                obs = obs[0]
            if reset_requested:
                got = env.reset()
                obs = got[0] if isinstance(got, tuple) else got
        if viser_viz is not None:
            try:
                viser_viz.update_from_env(base_env, rewards=rewards, actions=actions)
            except Exception as exc:
                print(f"[WARNING] Viser update failed: {exc}")
        if args_cli.real_time:
            sleep_t = dt - (time.time() - start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    env.close()
    if viser_viz is not None:
        viser_viz.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
