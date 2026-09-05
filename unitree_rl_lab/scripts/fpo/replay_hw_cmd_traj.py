#!/usr/bin/env python3
"""Replay exact hardware velocity commands in Isaac Lab and score sim↔real similarity.

Hardware logs (cmd_traj_run*.npz) provide cmd_vel + joint/IMU at 50 Hz.
This script pins the same cmd_vel timeline into Unitree-Go2-Velocity play env,
records sim joints (SDK order via joint_ids_map), and writes a similarity report.

Example:
  python scripts/fpo/replay_hw_cmd_traj.py --hw-npz .../cmd_traj_run1_....npz \\
      --sampling-steps 1 --headless --num_envs 1
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))

from isaaclab.app import AppLauncher
from isaaclab_fpo import cli_args  # isort: skip

import argparse
import json

parser = argparse.ArgumentParser(description="Replay HW cmd_vel in Isaac and compare.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--hw-npz", type=str, required=True, help="Hardware cmd_traj_run*.npz")
parser.add_argument("--sampling-steps", type=int, default=1, help="Must match HW ONNX (ours_s1 → 1).")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--joint-ids-map", type=str, default="3,0,9,6,4,1,10,7,5,2,11,8")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--viser", action="store_true", default=False)
parser.add_argument("--viser-port", type=int, default=8080)
parser.add_argument("--asset-dir", type=str, default=None)
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


def _pin_commands(env, cmds: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = cmds
    if hasattr(term, "time_left"):
        term.time_left[:] = 1.0e6


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def main() -> None:
    hw_path = pathlib.Path(args_cli.hw_npz).resolve()
    hw = np.load(hw_path, allow_pickle=True)
    hw_t = np.asarray(hw["t"], dtype=np.float64)
    hw_t = hw_t - hw_t[0]
    hw_cmd = np.asarray(hw["cmd_vel"], dtype=np.float64)
    hw_q = np.asarray(hw["q"], dtype=np.float64)
    hw_dq = np.asarray(hw["dq"], dtype=np.float64)
    hw_q_cmd = np.asarray(hw["q_cmd"], dtype=np.float64)
    hw_gyro = np.asarray(hw["gyro"], dtype=np.float64)
    joint_ids_map = [int(x) for x in args_cli.joint_ids_map.split(",") if x.strip() != ""]
    assert len(joint_ids_map) == 12

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_fpo_cfg(args_cli.task, args_cli)
    agent_cfg.policy.sampling_steps = int(args_cli.sampling_steps)

    log_root_path = os.path.abspath(os.path.join("logs", "fpo", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    out_dir = pathlib.Path(
        args_cli.out_dir
        or os.path.join(
            os.path.dirname(resume_path),
            "hw_sim2real",
            hw_path.stem,
            f"s{args_cli.sampling_steps}",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped

    viser_viz = None
    if args_cli.viser:
        from pathlib import Path as P

        from isaaclab_fpo.viser import ViserIsaacLab

        asset = (
            P(args_cli.asset_dir)
            if args_cli.asset_dir
            else _ROOT / "viser_assets" / "unitree_go2_velocity"
        )
        viser_viz = ViserIsaacLab(
            asset_dir=asset,
            port=int(args_cli.viser_port),
            update_freq=1,
            num_envs=1,
            env_spacing=0.0,
            fps=50,
        )
        viser_viz.load_from_env(base_env)
        print(f"[INFO] Viser at http://0.0.0.0:{args_cli.viser_port}")

    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] hw={hw_path}")
    print(f"[INFO] ckpt={resume_path} sampling_steps={args_cli.sampling_steps}")
    print(f"[INFO] duration={hw_t[-1]:.3f}s n_hw={len(hw_t)} out={out_dir}")

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.alg.policy.sampling_steps = int(args_cli.sampling_steps)
    if hasattr(runner.alg.policy, "_compiled_integrate_flow"):
        runner.alg.policy._compiled_integrate_flow = runner.alg.policy._integrate_flow
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    dt = float(env.unwrapped.step_dt)
    n_steps = int(np.floor(hw_t[-1] / dt)) + 1
    cmds = torch.zeros(n, 3, device=device)

    robot = env.unwrapped.scene["robot"]
    isaac_names = list(robot.data.joint_names)
    print(f"[INFO] isaac joints={isaac_names}")
    print(f"[INFO] joint_ids_map (isaac→sdk idx)={joint_ids_map}")

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    _pin_commands(env, cmds)

    hist = {k: [] for k in ("t", "cmd", "q_sdk", "dq_sdk", "gyro", "vel_b", "pos_w", "actions")}
    for step_i in range(n_steps):
        t = step_i * dt
        # nearest HW sample (HW & sim both ~50 Hz / 0.02 s)
        j = int(np.argmin(np.abs(hw_t - t)))
        cmds[0, 0] = float(hw_cmd[j, 0])
        cmds[0, 1] = float(hw_cmd[j, 1])
        cmds[0, 2] = float(hw_cmd[j, 2])
        _pin_commands(env, cmds)

        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]

        q_isaac = robot.data.joint_pos[0].detach().cpu().numpy()
        dq_isaac = robot.data.joint_vel[0].detach().cpu().numpy()
        # deploy.yaml joint_ids_map[isaac_i] = sdk_i  →  q_sdk[map] = q_isaac
        q_sdk = np.empty(12, dtype=np.float64)
        dq_sdk = np.empty(12, dtype=np.float64)
        q_sdk[joint_ids_map] = q_isaac
        dq_sdk[joint_ids_map] = dq_isaac
        gyro = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
        vel_b = np.concatenate(
            [
                robot.data.root_lin_vel_b[0, :2].detach().cpu().numpy(),
                robot.data.root_ang_vel_b[0, 2:3].detach().cpu().numpy(),
            ]
        )

        hist["t"].append(t)
        hist["cmd"].append(cmds[0].detach().cpu().numpy().copy())
        hist["q_sdk"].append(q_sdk.copy())
        hist["dq_sdk"].append(dq_sdk.copy())
        hist["gyro"].append(gyro.copy())
        hist["vel_b"].append(vel_b.copy())
        hist["pos_w"].append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
        hist["actions"].append(actions[0].detach().cpu().numpy().copy())

        if viser_viz is not None:
            try:
                viser_viz.update_from_env(base_env)
            except Exception as exc:
                print(f"[WARNING] Viser update failed: {exc}")

        if args_cli.real_time:
            sleep_t = dt - (time.time() - start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    env.close()
    if viser_viz is not None:
        viser_viz.close()

    sim_t = np.asarray(hist["t"])
    sim_cmd = np.stack(hist["cmd"])
    sim_q = np.stack(hist["q_sdk"])
    sim_dq = np.stack(hist["dq_sdk"])
    sim_gyro = np.stack(hist["gyro"])
    sim_vel = np.stack(hist["vel_b"])
    sim_pos = np.stack(hist["pos_w"])
    sim_act = np.stack(hist["actions"])

    # Align HW onto sim timeline by nearest neighbor
    idx = np.array([int(np.argmin(np.abs(hw_t - t))) for t in sim_t])
    hw_q_a = hw_q[idx]
    hw_dq_a = hw_dq[idx]
    hw_q_cmd_a = hw_q_cmd[idx]
    hw_gyro_a = hw_gyro[idx]
    hw_cmd_a = hw_cmd[idx]

    metrics = {
        "hw_npz": str(hw_path),
        "checkpoint": str(resume_path),
        "sampling_steps": int(args_cli.sampling_steps),
        "n_sim": int(len(sim_t)),
        "duration_s": float(sim_t[-1]),
        "cmd_mae": {
            "vx": _mae(sim_cmd[:, 0], hw_cmd_a[:, 0]),
            "vy": _mae(sim_cmd[:, 1], hw_cmd_a[:, 1]),
            "yaw": _mae(sim_cmd[:, 2], hw_cmd_a[:, 2]),
        },
        "joint_pos": {
            "corr": _corr(sim_q, hw_q_a),
            "mae": _mae(sim_q, hw_q_a),
            "rmse": _rmse(sim_q, hw_q_a),
            "corr_per_joint": [_corr(sim_q[:, i], hw_q_a[:, i]) for i in range(12)],
            "mae_per_joint": [_mae(sim_q[:, i], hw_q_a[:, i]) for i in range(12)],
        },
        "joint_vel": {
            "corr": _corr(sim_dq, hw_dq_a),
            "mae": _mae(sim_dq, hw_dq_a),
            "rmse": _rmse(sim_dq, hw_dq_a),
        },
        "gyro": {
            "corr": _corr(sim_gyro, hw_gyro_a),
            "mae": _mae(sim_gyro, hw_gyro_a),
            "rmse": _rmse(sim_gyro, hw_gyro_a),
        },
        # policy target on HW vs measured joint (sanity of HW tracking)
        "hw_q_vs_q_cmd": {
            "corr": _corr(hw_q_a, hw_q_cmd_a),
            "mae": _mae(hw_q_a, hw_q_cmd_a),
        },
        # sim measured joints vs HW policy target (cross domain)
        "sim_q_vs_hw_q_cmd": {
            "corr": _corr(sim_q, hw_q_cmd_a),
            "mae": _mae(sim_q, hw_q_cmd_a),
        },
        "sim_vel_tracking_mae": {
            "vx": _mae(sim_vel[:, 0], sim_cmd[:, 0]),
            "vy": _mae(sim_vel[:, 1], sim_cmd[:, 1]),
            "yaw": _mae(sim_vel[:, 2], sim_cmd[:, 2]),
        },
    }

    np.savez(
        out_dir / "sim_rollout.npz",
        t=sim_t,
        cmd=sim_cmd,
        q_sdk=sim_q,
        dq_sdk=sim_dq,
        gyro=sim_gyro,
        vel_b=sim_vel,
        pos_w=sim_pos,
        actions=sim_act,
        hw_q_aligned=hw_q_a,
        hw_dq_aligned=hw_dq_a,
        hw_q_cmd_aligned=hw_q_cmd_a,
        hw_gyro_aligned=hw_gyro_a,
        hw_cmd_aligned=hw_cmd_a,
        joint_ids_map=np.asarray(joint_ids_map, dtype=np.int32),
        sampling_steps=int(args_cli.sampling_steps),
        checkpoint=str(resume_path),
        hw_npz=str(hw_path),
    )
    with open(out_dir / "similarity.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    labels = ("vx", "vy", "yaw")
    for i, ax in enumerate(axes):
        ax.plot(sim_t, hw_cmd_a[:, i], "--", label=f"hw cmd {labels[i]}")
        ax.plot(sim_t, sim_cmd[:, i], label=f"sim cmd {labels[i]}", alpha=0.8)
        ax.plot(sim_t, sim_vel[:, i], label=f"sim vel {labels[i]}", alpha=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Cmd replay | {hw_path.stem} | s={args_cli.sampling_steps}")
    fig.savefig(out_dir / "cmd_tracking.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(sim_t, hw_q_a[:, i], "--", label="hw q", lw=1)
        ax.plot(sim_t, sim_q[:, i], label="sim q", lw=1)
        ax.set_title(f"j{i} corr={metrics['joint_pos']['corr_per_joint'][i]:.2f}", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Joint pos SDK-order | corr={metrics['joint_pos']['corr']:.3f} "
        f"mae={metrics['joint_pos']['mae']:.3f}"
    )
    fig.savefig(out_dir / "joint_pos_compare.png", dpi=140)
    plt.close(fig)

    print(
        f"[INFO] similarity joint_pos corr={metrics['joint_pos']['corr']:.4f} "
        f"mae={metrics['joint_pos']['mae']:.4f} | gyro corr={metrics['gyro']['corr']:.4f} "
        f"mae={metrics['gyro']['mae']:.4f}"
    )
    print(f"[INFO] wrote {out_dir}/similarity.json and plots")


if __name__ == "__main__":
    main()
    simulation_app.close()
