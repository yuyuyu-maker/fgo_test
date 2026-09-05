#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare PPO baseline vs FPO variants on official Unitree-Go2-Velocity play env."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))
sys.path.insert(0, str(_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="Play-env compare: PPO vs FPO on Unitree-Go2-Velocity.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument("--ppo_checkpoint", type=str, required=True)
parser.add_argument("--fpo_checkpoint", type=str, required=True)
parser.add_argument("--fpo_variant", type=str, default="all_ideas_teacher_kd")
parser.add_argument("--fpo_steps", type=int, nargs="+", default=[64, 8, 1])
parser.add_argument(
    "--fpo_plusplus_checkpoint",
    type=str,
    default=None,
    help="Optional FPO++ baseline checkpoint for side-by-side.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

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

from isaaclab_fpo import FpoRslRlVecEnvWrapper, cli_args as fpo_cli
from isaaclab_fpo.runners import OnPolicyRunner as FpoOnPolicyRunner


def make_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
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


def eval_ppo(ckpt: str) -> dict:
    print("\n>>> PPO baseline")
    env = make_env()
    # Build a minimal Namespace for rsl cfg
    ns = argparse.Namespace(
        seed=42,
        resume=False,
        load_run=None,
        checkpoint=ckpt,
        experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    agent_cfg = rsl_cli.parse_rsl_rl_cfg(args_cli.task, ns)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = RslOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(retrieve_file_path(ckpt))
    policy = runner.get_inference_policy(device=args_cli.device)

    def policy_fn(obs):
        return policy(obs)

    res = rollout(env, policy_fn, args_cli.device, args_cli.eval_episodes)
    print(
        f"  PPO  reward={res['mean_reward']:7.2f} ± {res['std_reward']:5.2f}  "
        f"len={res['mean_length']:6.1f}  (n={res['n']})"
    )
    env.close()
    return res


def eval_fpo(label: str, ckpt: str, variant: str, steps: list[int]) -> dict:
    print(f"\n>>> {label} ({variant})")
    env = make_env()
    ns = argparse.Namespace(
        seed=42,
        resume=False,
        load_run=None,
        checkpoint=ckpt,
        experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
        fpo_variant=variant,
        teacher_checkpoint=None,
    )
    agent_cfg = fpo_cli.parse_fpo_cfg(args_cli.task, ns)
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = FpoOnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    runner.load(retrieve_file_path(ckpt))
    runner.eval_mode()
    policy = runner.alg.policy
    policy._compiled_integrate_flow = policy._integrate_flow
    out = {}
    for s in steps:
        old = policy.sampling_steps
        policy.sampling_steps = s

        def policy_fn(obs, _s=s):
            norm = runner.obs_normalizer(obs) if runner.cfg.empirical_normalization else obs
            return policy.act_inference(norm, eval_mode="zero", eval_fixed_seed=12345)

        res = rollout(env, policy_fn, args_cli.device, args_cli.eval_episodes)
        out[s] = res
        print(
            f"  steps={s:3d}  reward={res['mean_reward']:7.2f} ± {res['std_reward']:5.2f}  "
            f"len={res['mean_length']:6.1f}  (n={res['n']})"
        )
        policy.sampling_steps = old
    env.close()
    return out


def main():
    print("=" * 80)
    print("Play-env compare on Unitree-Go2-Velocity")
    print(f"  envs={args_cli.num_envs}  episodes={args_cli.eval_episodes}  device={args_cli.device}")
    print("=" * 80)
    ppo = eval_ppo(args_cli.ppo_checkpoint)
    fpo = eval_fpo("all_ideas+PPO-KD", args_cli.fpo_checkpoint, args_cli.fpo_variant, args_cli.fpo_steps)
    fpo_pp = None
    if args_cli.fpo_plusplus_checkpoint:
        fpo_pp = eval_fpo("FPO++ baseline", args_cli.fpo_plusplus_checkpoint, "baseline", args_cli.fpo_steps)

    print("\n" + "=" * 80)
    print("SUMMARY (mean reward on play env, cmd=full ranges)")
    print(f"{'model':<28}" + "".join(f"{s:>10}" for s in ["—"] + args_cli.fpo_steps))
    print(f"{'PPO baseline':<28}{ppo['mean_reward']:10.2f}" + "".join(f"{'—':>10}" for _ in args_cli.fpo_steps))
    row = f"{'all_ideas PPO-KD':<28}{'—':>10}"
    for s in args_cli.fpo_steps:
        row += f"{fpo[s]['mean_reward']:10.2f}"
    print(row)
    if fpo_pp is not None:
        row = f"{'FPO++ baseline':<28}{'—':>10}"
        for s in args_cli.fpo_steps:
            row += f"{fpo_pp[s]['mean_reward']:10.2f}"
        print(row)
    print("=" * 80)
    # deltas vs PPO
    print("\nDelta vs PPO (all_ideas):")
    for s in args_cli.fpo_steps:
        d = fpo[s]["mean_reward"] - ppo["mean_reward"]
        print(f"  steps={s:3d}: {d:+.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
