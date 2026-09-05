#!/usr/bin/env python3
"""Pinned-command tracking plots for official Unitree-Go2-Velocity (PPO or FPO)."""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))
sys.path.insert(0, str(_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=("ppo", "fpo"), required=True)
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--title", type=str, default="")
parser.add_argument("--flow-sampling-steps", type=int, default=None)
from isaaclab_fpo import cli_args as fpo_cli
import cli_args as rsl_cli

fpo_cli.add_fpo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not getattr(args_cli, "checkpoint", None):
    parser.error("--checkpoint is required")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import os
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper, cli_args as fpo_cli_mod
from isaaclab_fpo.runners import OnPolicyRunner as FpoOnPolicyRunner


def _pin_commands(env, cmds: torch.Tensor):
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = cmds
    if hasattr(term, "time_left"):
        term.time_left[:] = 1.0e6


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    os.makedirs(args_cli.out_dir, exist_ok=True)

    if args_cli.kind == "ppo":
        agent_cfg = rsl_cli.parse_rsl_rl_cfg(args_cli.task, args_cli)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = RslOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        agent_cfg = fpo_cli_mod.parse_fpo_cfg(args_cli.task, args_cli)
        env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = FpoOnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        if args_cli.flow_sampling_steps is not None:
            runner.alg.policy.sampling_steps = int(args_cli.flow_sampling_steps)
            runner.alg.policy._compiled_integrate_flow = runner.alg.policy._integrate_flow
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    n = env.unwrapped.num_envs
    cmds = torch.zeros(n, 3, device=env.unwrapped.device)
    presets = [
        (0.80, 0.00, 0.00),
        (0.50, 0.00, 0.60),
        (0.00, 0.30, 0.00),
        (0.00, 0.00, 0.00),
    ]
    for i in range(n):
        cmds[i] = torch.tensor(presets[i % len(presets)], device=cmds.device)

    obs, _ = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    hist = {k: [] for k in ("t", "cmd", "vel_b", "pos_w")}
    for step_i in range(args_cli.steps):
        _pin_commands(env, cmds)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        hist["t"].append(step_i * env.unwrapped.step_dt)
        hist["cmd"].append(cmds.detach().cpu().numpy())
        lin_xy = robot.data.root_lin_vel_b[:, :2].detach().cpu().numpy()
        yaw = robot.data.root_ang_vel_b[:, 2:3].detach().cpu().numpy()
        hist["vel_b"].append(np.concatenate([lin_xy, yaw], axis=-1))
        hist["pos_w"].append(robot.data.root_pos_w.detach().cpu().numpy())

    env.close()

    t = np.array(hist["t"])
    cmd = np.stack(hist["cmd"])
    vel = np.stack(hist["vel_b"])
    pos = np.stack(hist["pos_w"])
    np.savez(os.path.join(args_cli.out_dir, "rollout.npz"), t=t, cmd=cmd, vel=vel, pos=pos)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["forward 0.8 m/s", "forward 0.5 + yaw 0.6", "sidestep 0.3 m/s", "stand"]
    title = args_cli.title or args_cli.kind
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes.ravel()
    for i in range(min(n, 4)):
        ax[i].plot(t, cmd[:, i, 0], "--", label="cmd vx", color="C0")
        ax[i].plot(t, vel[:, i, 0], label="actual vx", color="C0")
        ax[i].plot(t, cmd[:, i, 2], "--", label="cmd wz", color="C1")
        ax[i].plot(t, vel[:, i, 2], label="actual wz", color="C1")
        ax[i].set_title(labels[i])
        ax[i].legend(fontsize=8)
        ax[i].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.savefig(os.path.join(args_cli.out_dir, "velocity_tracking.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(min(n, 4)):
        ax.plot(pos[:, i, 0] - pos[0, i, 0], pos[:, i, 1] - pos[0, i, 1], label=labels[i])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-down trajectory")
    ax.legend()
    fig.savefig(os.path.join(args_cli.out_dir, "trajectory_xy.png"), dpi=140)
    plt.close(fig)

    vx_err = float(np.mean(np.abs(vel[:, 0, 0] - cmd[:, 0, 0])))
    fallen = float(np.mean(pos[:, 0, 2] < 0.2))
    summary = f"env0 |vx-cmd|={vx_err:.3f}  frac_z<0.2={fallen:.3f}  title={title}\n"
    (pathlib.Path(args_cli.out_dir) / "tracking_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
    simulation_app.close()
