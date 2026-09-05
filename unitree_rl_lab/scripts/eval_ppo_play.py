#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play-env mean reward for a Gaussian PPO checkpoint (locomotion main table)."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))
sys.path.insert(0, str(_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="PPO play-env mean reward.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import json
import numpy as np
import torch
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import cli_args as rsl_cli


def make_env():
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            entry_point_key="play_env_cfg_entry_point",
        )
    except Exception:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return env


def rollout(env, policy_fn, device, eval_episodes: int) -> dict:
    obs, _ = env.get_observations()
    obs = obs.to(device)
    episode_reward = torch.zeros(env.num_envs, device=device)
    episode_length = torch.zeros(env.num_envs, device=device)
    episodes_per_env = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    target = max(1, eval_episodes // env.num_envs)
    max_len = getattr(env, "max_episode_length", 1000)
    rewards, lengths = [], []
    while (episodes_per_env < target).any():
        with torch.no_grad():
            actions = policy_fn(obs)
        obs, rew, dones, _ = env.step(actions.to(env.device))
        obs = obs.to(device)
        episode_reward += rew.to(device)
        episode_length += 1
        done_mask = (dones > 0) | (episode_length >= max_len)
        if done_mask.any():
            for idx in done_mask.nonzero(as_tuple=False).squeeze(-1):
                if episodes_per_env[idx] < target:
                    rewards.append(episode_reward[idx].item())
                    lengths.append(episode_length[idx].item())
                    episodes_per_env[idx] += 1
            episode_reward[done_mask] = 0
            episode_length[done_mask] = 0
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n": len(rewards),
    }


def main():
    env = make_env()
    ns = argparse.Namespace(
        seed=42,
        resume=False,
        load_run=None,
        checkpoint=args_cli.checkpoint,
        experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    agent_cfg = rsl_cli.parse_rsl_rl_cfg(args_cli.task, ns)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = RslOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=args_cli.device)

    def policy_fn(obs):
        return policy(obs)

    res = rollout(env, policy_fn, args_cli.device, args_cli.eval_episodes)
    print(
        f"PPO play  task={args_cli.task}  reward={res['mean_reward']:7.2f} ± {res['std_reward']:5.2f}  "
        f"len={res['mean_length']:6.1f}  (n={res['n']})"
    )
    print("SUMMARY_JSON " + json.dumps({"task": args_cli.task, "algo": "ppo", **res, "checkpoint": args_cli.checkpoint}))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
