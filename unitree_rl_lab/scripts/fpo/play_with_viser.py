#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play official Unitree FPO policy with Viser web visualization.

Policy / env stay on Unitree-Go2-Velocity (45-D). Only the viewer differs from play.py.

Example:
  python scripts/fpo/play_with_viser.py --task Unitree-Go2-Velocity --headless \\
    --fpo_variant all_ideas_teacher_kd \\
    --checkpoint logs/fpo/.../model_1499.pt \\
    --num_envs 16 --viser --viser-port 8080 \\
    --flow-sampling-steps 64
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))

# Import viser before Isaac / websockets stack (same HACK as isaaclab_experiments).
import viser  # noqa: F401

from isaaclab.app import AppLauncher
from isaaclab_fpo import cli_args  # isort: skip

import argparse

parser = argparse.ArgumentParser(description="Play Unitree FPO with Viser visualization.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--viser", action="store_true", default=True, help="Enable Viser (default on).")
parser.add_argument("--no-viser", action="store_false", dest="viser")
parser.add_argument("--viser-port", type=int, default=8080)
parser.add_argument("--asset-dir", type=str, default=None)
parser.add_argument("--viser-update-freq", type=int, default=1)
parser.add_argument("--viser-env-spacing", type=float, default=1.5)
parser.add_argument("--viser-fps", type=int, default=60)
parser.add_argument("--viser-random-grid-size", type=float, default=0.0)
parser.add_argument(
    "--flow-sampling-steps",
    type=int,
    default=None,
    help="Override Euler steps at inference (default: training value, usually 64).",
)
parser.add_argument(
    "--square-walk",
    action="store_true",
    default=False,
    help="Pin body-frame velocity cmds: fwd/left/back/right cycle (overrides env command sampler).",
)
parser.add_argument("--square-speed", type=float, default=0.3, help="Square-walk linear speed (m/s).")
parser.add_argument("--square-segment-s", type=float, default=3.0, help="Seconds per square side.")
parser.add_argument("--square-warmup-s", type=float, default=1.0, help="Standstill before first side.")
parser.add_argument(
    "--cmd-npz",
    type=str,
    default=None,
    help="Replay exact HW cmd_vel timeline from cmd_traj_run*.npz (overrides square-walk / env cmds).",
)
parser.add_argument("--cmd-loop", action="store_true", default=True, help="Loop cmd-npz when finished.")
parser.add_argument("--no-cmd-loop", action="store_false", dest="cmd_loop")
cli_args.add_fpo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import os
import time

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper
from isaaclab_fpo.runners import OnPolicyRunner
from isaaclab_fpo.viser import ViserIsaacLab

_SQUARE_DIRS = (
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


def _square_cmd_at(t: float, *, speed: float, segment_s: float, warmup_s: float) -> tuple[float, float, float, str]:
    if t < warmup_s:
        return 0.0, 0.0, 0.0, "warmup"
    u = t - warmup_s
    idx = int(u // segment_s) % 4
    name, sx, sy, sz = _SQUARE_DIRS[idx]
    return speed * sx, speed * sy, speed * sz, name


def _resolve_asset_dir(task: str, asset_dir_arg: str | None) -> Path:
    if asset_dir_arg:
        return Path(asset_dir_arg)
    task_clean = task.lower().replace(":", "_").replace("-", "_")
    candidates = [
        _ROOT / "viser_assets" / task_clean,
        _ROOT / "viser_assets" / "unitree_go2_velocity",
        Path("/workspace/fgo_test/isaaclab_experiments/isaaclab_fpo/viser_assets/isaac_velocity_flat_unitree_go2_v0"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Viser asset dir not found. Expected under unitree_rl_lab/viser_assets/ "
        f"(tried {[str(c) for c in candidates]})."
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_fpo_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "fpo", agent_cfg.experiment_name))
    print(f"[INFO] Loading FPO experiment from {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped
    viser_viz = None
    if args_cli.viser:
        asset_dir = _resolve_asset_dir(args_cli.task, args_cli.asset_dir)
        print(f"[INFO] Viser assets: {asset_dir}")
        num_envs_to_viz = min(env_cfg.scene.num_envs, 64)
        random_offsets = None
        if args_cli.viser_random_grid_size > 0:
            half = args_cli.viser_random_grid_size / 2
            random_offsets = np.random.uniform(
                low=[-half, -half, 0],
                high=[half, half, 0],
                size=(num_envs_to_viz, 3),
            )
            env_spacing = 0.0
        else:
            env_spacing = args_cli.viser_env_spacing
        try:
            viser_viz = ViserIsaacLab(
                asset_dir=asset_dir,
                port=args_cli.viser_port,
                update_freq=args_cli.viser_update_freq,
                num_envs=num_envs_to_viz,
                env_spacing=env_spacing,
                fps=args_cli.viser_fps,
                random_offsets=random_offsets,
            )
            viser_viz.load_from_env(base_env)
            print(f"[INFO] Viser at http://0.0.0.0:{args_cli.viser_port} (forward with ssh -L)")
        except Exception as exc:
            print(f"[ERROR] Viser init failed: {exc}")
            import traceback

            traceback.print_exc()
            viser_viz = None

    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    if args_cli.flow_sampling_steps is not None and hasattr(runner.alg.policy, "sampling_steps"):
        runner.alg.policy.sampling_steps = int(args_cli.flow_sampling_steps)
        print(f"[INFO] Inference sampling_steps={runner.alg.policy.sampling_steps}")
    # Disable torch.compile path for interactive play (avoids Dynamo issues).
    if hasattr(runner.alg.policy, "_compiled_integrate_flow"):
        runner.alg.policy._compiled_integrate_flow = runner.alg.policy._integrate_flow

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    timestep = 0
    sim_t = 0.0
    last_phase = None
    cmds = None
    hw_t = None
    hw_cmd = None
    if args_cli.cmd_npz:
        hw = np.load(args_cli.cmd_npz, allow_pickle=True)
        hw_t = np.asarray(hw["t"], dtype=np.float64)
        hw_t = hw_t - hw_t[0]
        hw_cmd = np.asarray(hw["cmd_vel"], dtype=np.float64)
        n = env.unwrapped.num_envs
        cmds = torch.zeros(n, 3, device=env.unwrapped.device)
        print(
            f"[INFO] HW cmd replay ON: {args_cli.cmd_npz} "
            f"duration={hw_t[-1]:.2f}s loop={args_cli.cmd_loop}"
        )
        _pin_commands(env, cmds)
    elif args_cli.square_walk:
        n = env.unwrapped.num_envs
        cmds = torch.zeros(n, 3, device=env.unwrapped.device)
        print(
            f"[INFO] Square walk ON: |v|={args_cli.square_speed} m/s, "
            f"segment={args_cli.square_segment_s}s, warmup={args_cli.square_warmup_s}s "
            f"(fwd→left→back→right, looping)"
        )
        _pin_commands(env, cmds)

    while simulation_app.is_running():
        start_time = time.time()
        reset_requested = viser_viz is not None and viser_viz.check_reset_request()
        if reset_requested:
            sim_t = 0.0
            last_phase = None
        if hw_cmd is not None and cmds is not None:
            t_query = float(sim_t)
            if args_cli.cmd_loop and hw_t[-1] > 0:
                t_query = t_query % float(hw_t[-1])
            j = int(np.argmin(np.abs(hw_t - t_query)))
            vx, vy, yaw = float(hw_cmd[j, 0]), float(hw_cmd[j, 1]), float(hw_cmd[j, 2])
            cmds[:, 0], cmds[:, 1], cmds[:, 2] = vx, vy, yaw
            _pin_commands(env, cmds)
        elif args_cli.square_walk and cmds is not None:
            vx, vy, yaw, phase = _square_cmd_at(
                sim_t,
                speed=float(args_cli.square_speed),
                segment_s=float(args_cli.square_segment_s),
                warmup_s=float(args_cli.square_warmup_s),
            )
            cmds[:, 0], cmds[:, 1], cmds[:, 2] = vx, vy, yaw
            _pin_commands(env, cmds)
            if phase != last_phase:
                print(f"[INFO] t={sim_t:.1f}s phase={phase} cmd=({vx:.2f},{vy:.2f},{yaw:.2f})")
                last_phase = phase
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            if reset_requested:
                obs, _ = env.reset()
                if cmds is not None:
                    cmds.zero_()
                    _pin_commands(env, cmds)
        sim_t += float(dt)
        if viser_viz is not None:
            try:
                viser_viz.update_from_env(base_env, rewards=rewards, actions=actions)
            except Exception as exc:
                print(f"[WARNING] Viser update failed: {exc}")
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()
    if viser_viz is not None:
        viser_viz.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
