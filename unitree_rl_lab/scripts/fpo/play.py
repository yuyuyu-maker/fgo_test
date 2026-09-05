# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play / export a flow-policy checkpoint trained on official Unitree tasks.

Default --fpo_variant is FPO++ (baseline). Other variants are research ideas, not FPO++.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))

from isaaclab.app import AppLauncher
from isaaclab_fpo import cli_args  # isort: skip

import argparse

parser = argparse.ArgumentParser(description="Play FPO++ (baseline) or an ideas variant on a Unitree RL Lab task.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--export-only",
    action="store_true",
    help="Load checkpoint, export ONNX/JIT, then exit (no sim loop).",
)
cli_args.add_fpo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_fpo.runners import OnPolicyRunner


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

    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(policy_nn, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    except Exception as err:
        print(f"[WARN] JIT export skipped ({err.__class__.__name__})")
    export_policy_as_onnx(policy_nn, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")
    print(f"[INFO] Exported {export_model_dir}/policy.onnx")
    if args_cli.export_only:
        env.close()
        return

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
lose()
