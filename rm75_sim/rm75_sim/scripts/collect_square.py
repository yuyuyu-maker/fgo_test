#!/usr/bin/env python3
"""Square env smoke + optional few-episode skeleton collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rm75_sim.dataset.writer import SimLeRobotWriter
from rm75_sim.envs.square_env import SquareEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, default=0, help="If >0, save N hold-at-home episodes (skeleton)")
    ap.add_argument("--repo-id", type=str, default="local/rm75_sim_square_v0")
    ap.add_argument("--root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = SquareEnv(gui=False, seed=args.seed)
    obs = env.reset()
    assert obs["observation.state"].shape == (14,)
    print("Square reset OK; is_success=", env.is_success())
    # random small motions should not succeed
    for _ in range(5):
        a = obs["observation.state"].copy()
        a[:7] += np.random.uniform(-0.01, 0.01, size=7).astype(np.float32)
        obs, r, done, info = env.step(a)
    print("after jitter is_success=", env.is_success(), "info=", info)

    # Place block into slot by teleport to verify success metric
    c = env.cfg
    sx, sy = c.slot_center
    import pybullet as p

    he = [s / 2 for s in c.block_size]
    p.resetBasePositionAndOrientation(
        env.block_id,
        [sx, sy, c.table_height + he[2] * 0.6],
        p.getQuaternionFromEuler([0, 0, 0]),
    )
    for _ in range(20):
        p.stepSimulation()
    print("teleport-in-slot is_success=", env.is_success())

    if args.collect > 0:
        writer = SimLeRobotWriter(
            repo_id=args.repo_id,
            root=args.root,
            task="insert the yellow square block into the narrow blue slot",
            image_hw=env.cfg.image_size,
            camera_names=env.cfg.camera_names,
        )
        for i in range(args.collect):
            obs = env.reset(seed=args.seed + i)
            frames = [obs]
            hold = obs["observation.state"]
            for _ in range(30):
                obs, _, _, _ = env.step(hold)
                frames.append(obs)
            writer.add_episode(frames)
            print(f"square skeleton episode {i+1} saved")
        print("dataset:", writer.dataset_path)

    env.close()
    print("Square smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
