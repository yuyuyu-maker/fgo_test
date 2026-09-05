#!/usr/bin/env python3
"""Go2 Ours square-velocity walk in Isaac Lab for sim↔real similarity logs.

Command cycle (body frame), each SEGMENT_S seconds:
  forward  +vx | left +vy | backward -vx | right -vy | (repeat)

Dumps rollout.npz with t, cmd, vel_b, pos_w, actions — use the same schedule on hardware.

Example:
  python scripts/fpo/play_go2_square_walk.py --checkpoint .../model_1499.pt \\
      --sampling-steps 64 --headless --num_envs 1 --cycles 2
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))

from isaaclab.app import AppLauncher
from isaaclab_fpo import cli_args  # isort: skip

import argparse

parser = argparse.ArgumentParser(description="Go2 square-velocity walk (Ours / FPO).")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--speed", type=float, default=0.3, help="Linear speed m/s for each leg.")
parser.add_argument("--segment_s", type=float, default=3.0, help="Seconds per direction.")
parser.add_argument("--cycles", type=int, default=3, help="How many full square cycles.")
parser.add_argument("--warmup_s", type=float, default=1.0, help="Stand still before first move.")
parser.add_argument("--sampling-steps", type=int, default=64)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_fpo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.fpo_variant = args_cli.fpo_variant or "all_ideas_teacher_kd"
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper
from isaaclab_fpo.runners import OnPolicyRunner


DIRECTIONS = (
    ("forward", +1.0, 0.0, 0.0),
    ("left", 0.0, +1.0, 0.0),
    ("backward", -1.0, 0.0, 0.0),
    ("right", 0.0, -1.0, 0.0),
)


def _pin_commands(env, cmds: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = cmds
    if hasattr(term, "time_left"):
        term.time_left[:] = 1.0e6


def _cmd_at(t: float, *, speed: float, segment_s: float, warmup_s: float) -> tuple[float, float, float, str]:
    if t < warmup_s:
        return 0.0, 0.0, 0.0, "warmup"
    u = t - warmup_s
    idx = int(u // segment_s) % 4
    name, sx, sy, sz = DIRECTIONS[idx]
    return speed * sx, speed * sy, speed * sz, name


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_fpo_cfg(args_cli.task, args_cli)
    # Force play sampling steps (override cfg default 64).
    agent_cfg.policy.sampling_steps = int(args_cli.sampling_steps)

    log_root_path = os.path.abspath(os.path.join("logs", "fpo", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    total_s = float(args_cli.warmup_s) + float(args_cli.cycles) * 4.0 * float(args_cli.segment_s)
    out_dir = args_cli.out_dir or os.path.join(
        log_dir, "square_walk", f"s{args_cli.sampling_steps}_v{args_cli.speed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(out_dir, "videos"),
            step_trigger=lambda step: step == 0,
            video_length=int(total_s / env_cfg.sim.dt / env_cfg.decimation) + 10,
            disable_logger=True,
            name_prefix=f"square_s{args_cli.sampling_steps}",
        )
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] ckpt={resume_path}")
    print(f"[INFO] sampling_steps={args_cli.sampling_steps} speed={args_cli.speed} segment_s={args_cli.segment_s}")
    print(f"[INFO] total_s≈{total_s:.1f} out={out_dir}")

    # FPO OnPolicyRunner expects the dataclass (train_cfg.policy), not a plain dict.
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    # Ensure runtime steps match CLI (runner may have restored cfg).
    runner.alg.policy.sampling_steps = int(args_cli.sampling_steps)
    if hasattr(runner.alg.policy, "_compiled_integrate_flow"):
        runner.alg.policy._compiled_integrate_flow = runner.alg.policy._integrate_flow
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    dt = float(env.unwrapped.step_dt)
    n_steps = int(np.ceil(total_s / dt))
    cmds = torch.zeros(n, 3, device=device)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    robot = env.unwrapped.scene["robot"]

    hist = {k: [] for k in ("t", "cmd", "vel_b", "pos_w", "phase", "actions")}
    _pin_commands(env, cmds)

    for step_i in range(n_steps):
        t = step_i * dt
        vx, vy, yaw, phase = _cmd_at(
            t, speed=float(args_cli.speed), segment_s=float(args_cli.segment_s), warmup_s=float(args_cli.warmup_s)
        )
        cmds[:, 0], cmds[:, 1], cmds[:, 2] = vx, vy, yaw
        _pin_commands(env, cmds)

        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]

        hist["t"].append(t)
        hist["cmd"].append(cmds.detach().cpu().numpy().copy())
        lin_xy = robot.data.root_lin_vel_b[:, :2].detach().cpu().numpy()
        wz = robot.data.root_ang_vel_b[:, 2:3].detach().cpu().numpy()
        hist["vel_b"].append(np.concatenate([lin_xy, wz], axis=-1))
        hist["pos_w"].append(robot.data.root_pos_w.detach().cpu().numpy())
        hist["phase"].append(phase)
        hist["actions"].append(actions.detach().cpu().numpy())

        if args_cli.real_time:
            sleep_t = dt - (time.time() - start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    env.close()

    t = np.asarray(hist["t"], dtype=np.float64)
    cmd = np.stack(hist["cmd"])
    vel = np.stack(hist["vel_b"])
    pos = np.stack(hist["pos_w"])
    actions = np.stack(hist["actions"])
    phases = np.asarray(hist["phase"])

    np.savez(
        os.path.join(out_dir, "rollout.npz"),
        t=t,
        cmd=cmd,
        vel_b=vel,
        pos_w=pos,
        actions=actions,
        phases=phases,
        speed=float(args_cli.speed),
        segment_s=float(args_cli.segment_s),
        sampling_steps=int(args_cli.sampling_steps),
        checkpoint=str(resume_path),
    )

    # Schedule sidecar for hardware replay
    schedule_path = os.path.join(out_dir, "cmd_schedule.csv")
    with open(schedule_path, "w", encoding="utf-8") as f:
        f.write("t,vx,vy,yaw,phase\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.4f},{cmd[i, 0, 0]:.4f},{cmd[i, 0, 1]:.4f},{cmd[i, 0, 2]:.4f},{phases[i]}\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, cmd[:, 0, 0], "--", label="cmd vx")
    axes[0].plot(t, vel[:, 0, 0], label="actual vx")
    axes[0].set_ylabel("vx (m/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, cmd[:, 0, 1], "--", label="cmd vy")
    axes[1].plot(t, vel[:, 0, 1], label="actual vy")
    axes[1].set_ylabel("vy (m/s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, pos[:, 0, 0] - pos[0, 0, 0], label="Δx")
    axes[2].plot(t, pos[:, 0, 1] - pos[0, 0, 1], label="Δy")
    axes[2].set_xlabel("t (s)")
    axes[2].set_ylabel("world Δ (m)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Go2 Ours square walk | steps={args_cli.sampling_steps} | |v|={args_cli.speed}")
    fig.savefig(os.path.join(out_dir, "square_tracking.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos[:, 0, 0] - pos[0, 0, 0], pos[:, 0, 1] - pos[0, 0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx (m)")
    ax.set_ylabel("Δy (m)")
    ax.set_title("Top-down trajectory")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "trajectory_xy.png"), dpi=140)
    plt.close(fig)

    vx_err = float(np.mean(np.abs(vel[:, 0, 0] - cmd[:, 0, 0])))
    vy_err = float(np.mean(np.abs(vel[:, 0, 1] - cmd[:, 0, 1])))
    print(f"[INFO] mean |vx-cmd|={vx_err:.3f} |vy-cmd|={vy_err:.3f}")
    print(f"[INFO] wrote {out_dir}/rollout.npz and cmd_schedule.csv")


if __name__ == "__main__":
    main()
    simulation_app.close()
