#!/usr/bin/env python3
"""Collect Can demos with script expert → LeRobot dataset (separate from Square)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rm75_sim.dataset.writer import SimLeRobotWriter
from rm75_sim.envs.can_env import CanEnv
from rm75_sim.expert.can_script import CanScriptExpert


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-success", type=int, default=5)
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-id", type=str, default="local/rm75_sim_can_v0")
    ap.add_argument(
        "--root",
        type=str,
        default=str(ROOT / "data"),
    )
    ap.add_argument("--task", type=str, default="place the red cylinder into the green bin")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    env = CanEnv(gui=args.gui, seed=args.seed)
    expert = CanScriptExpert(env)
    writer = SimLeRobotWriter(
        repo_id=args.repo_id,
        root=args.root,
        task=args.task,
        image_hw=env.cfg.image_size,
        camera_names=env.cfg.camera_names,
    )

    n_ok = 0
    attempt = 0
    while n_ok < args.num_success and attempt < args.max_attempts:
        seed = args.seed + attempt
        attempt += 1
        frames, success, meta = expert.rollout(seed=seed, render=True)
        print(
            f"attempt={attempt} seed={seed} success={success} "
            f"frames={meta['n_frames']} is_success_now={env.is_success()}"
        )
        if not success:
            continue
        n = writer.add_episode(frames)
        n_ok += 1
        print(f"  saved episode {n_ok}/{args.num_success} ({n} frames) -> {writer.dataset_path}")

    env.close()
    if n_ok < args.num_success:
        print(f"FAILED: only {n_ok}/{args.num_success} successful episodes")
        return 1
    print("DONE", writer.dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
