"""Replay the trained Unitree Go2 PPO policy and dump tracking plots (+ optional video)."""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--video", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


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
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    out_dir = args_cli.out_dir or os.path.join(log_dir, "demo")
    os.makedirs(out_dir, exist_ok=True)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(out_dir, "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.steps,
            disable_logger=True,
            name_prefix="go2_ppo",
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
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
    hist = {k: [] for k in ("t", "cmd", "vel_b", "pos_w", "rpy_grav_z")}
    _pin_commands(env, cmds)

    frames = []
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
        hist["rpy_grav_z"].append(robot.data.projected_gravity_b[:, 2].detach().cpu().numpy())

        if args_cli.video and step_i in (0, args_cli.steps // 2, args_cli.steps - 1):
            try:
                rgb = env.unwrapped.render()
                if rgb is not None:
                    frames.append(rgb)
            except Exception as exc:
                print(f"[WARN] render failed at step {step_i}: {exc}")

    env.close()

    t = np.array(hist["t"])
    cmd = np.stack(hist["cmd"])
    vel = np.stack(hist["vel_b"])
    pos = np.stack(hist["pos_w"])
    np.savez(os.path.join(out_dir, "rollout.npz"), t=t, cmd=cmd, vel=vel, pos=pos)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["forward 0.8 m/s", "forward 0.5 + yaw 0.6", "sidestep 0.3 m/s", "stand"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes.ravel()
    for i in range(min(n, 4)):
        ax[i].plot(t, cmd[:, i, 0], "--", label="cmd vx", color="C0")
        ax[i].plot(t, vel[:, i, 0], label="actual vx", color="C0")
        ax[i].plot(t, cmd[:, i, 2], "--", label="cmd wz", color="C1")
        ax[i].plot(t, vel[:, i, 2], label="actual wz", color="C1")
        ax[i].set_title(labels[i])
        ax[i].set_xlabel("time (s)")
        ax[i].set_ylabel("m/s or rad/s")
        ax[i].legend(fontsize=8)
        ax[i].grid(True, alpha=0.3)
    fig.suptitle("Unitree Go2 official PPO (3000 iter) — velocity tracking")
    fig.savefig(os.path.join(out_dir, "velocity_tracking.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(min(n, 4)):
        ax.plot(pos[:, i, 0] - pos[0, i, 0], pos[:, i, 1] - pos[0, i, 1], label=labels[i])
        ax.scatter(pos[-1, i, 0] - pos[0, i, 0], pos[-1, i, 1] - pos[0, i, 1], marker="o")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x displacement (m)")
    ax.set_ylabel("y displacement (m)")
    ax.set_title("Top-down trajectory (10 s rollout)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "trajectory_xy.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(n, 4)):
        ax.plot(t, pos[:, i, 2], label=labels[i])
    ax.axhline(0.3, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("base height z (m)")
    ax.set_title("Base height (should stay ~0.3-0.4 m if not fallen)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "base_height.png"), dpi=140)
    plt.close(fig)

    vx_err = np.mean(np.abs(vel[:, 0, 0] - cmd[:, 0, 0]))
    dist = np.linalg.norm(pos[-1, 0, :2] - pos[0, 0, :2])
    print("[INFO] env0 |vx-cmd| mean={:.3f} m/s, xy distance={:.2f} m in {:.1f}s".format(vx_err, dist, t[-1]))
    print("[INFO] plots: {}".format(out_dir))
    if frames:
        from PIL import Image

        for k, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(out_dir, "frame_{}.png".format(k)))
        print("[INFO] saved {} rgb frames".format(len(frames)))


if __name__ == "__main__":
    main()
    simulation_app.close()
