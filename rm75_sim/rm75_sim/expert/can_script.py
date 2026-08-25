"""Scripted Can expert with soft-grasp constraint (reliable demos under approx gripper)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.can_env import CanEnv


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    return [a * (1 - t) + b * t for t in np.linspace(0, 1, n)]


class CanScriptExpert:
    def __init__(self, env: CanEnv, steps_per_seg: int = 20):
        self.env = env
        self.steps_per_seg = steps_per_seg

    def compute_joint_path(self) -> tuple[list[tuple[np.ndarray, float]], int, int]:
        """Returns path, grasp_idx (close+attach), release_idx (open+detach)."""
        env = self.env
        c = env.cfg
        obj_pos, _ = env.object_pose()
        bx, by = c.bin_center

        # Start from a reachable stretch seed near workspace
        seed = np.deg2rad([0.0, 25.0, 0.0, 70.0, 0.0, 70.0, 0.0])
        env.robot.set_arm_joints(seed, reset=True)
        env.robot.set_gripper_open(1.0, reset=True)

        hover_z = c.table_height + c.object_height + 0.10
        grasp_z = c.table_height + c.object_height * 0.65
        place_hover_z = c.table_height + c.bin_height + 0.12
        place_z = c.table_height + c.object_height * 0.7

        phases = [
            ("move", np.array([obj_pos[0], obj_pos[1], hover_z]), 1.0),
            ("move", np.array([obj_pos[0], obj_pos[1], grasp_z]), 1.0),
            ("grasp", np.array([obj_pos[0], obj_pos[1], grasp_z]), 0.0),
            ("move", np.array([obj_pos[0], obj_pos[1], hover_z]), 0.0),
            ("move", np.array([bx, by, place_hover_z]), 0.0),
            ("move", np.array([bx, by, place_z]), 0.0),
            ("release", np.array([bx, by, place_z]), 1.0),
            ("move", np.array([bx, by, place_hover_z]), 1.0),
        ]

        q_cur = env.robot.get_arm_joints()
        g_cur = 1.0
        path: list[tuple[np.ndarray, float]] = []
        grasp_idx = release_idx = -1

        for kind, target_pos, g_tgt in phases:
            q_tgt = env.robot.ik(target_pos, position_only=True, n_restarts=4)
            env.robot.set_arm_joints(q_tgt, reset=True)
            pos, _ = env.robot.get_eef_pose()
            if np.linalg.norm(pos - target_pos) > 0.03:
                raise RuntimeError(
                    f"IK failed: target={target_pos} got={pos} "
                    f"err={np.linalg.norm(pos - target_pos):.3f}"
                )
            for q, g in zip(
                _lerp(q_cur, q_tgt, self.steps_per_seg),
                _lerp(np.array([g_cur]), np.array([g_tgt]), self.steps_per_seg),
            ):
                path.append((q.astype(np.float64), float(g[0])))
            if kind == "grasp":
                grasp_idx = len(path) - 1
            if kind == "release":
                release_idx = len(path) - 1
            q_cur, g_cur = q_tgt, g_tgt

        for _ in range(20):
            path.append((q_cur.copy(), g_cur))
        return path, grasp_idx, release_idx

    def rollout(
        self, seed: Optional[int] = None, render: bool = True
    ) -> tuple[list[dict], bool, dict]:
        self.env.reset(seed=seed)
        path, grasp_idx, release_idx = self.compute_joint_path()
        obs = self.env.reset(seed=seed)
        # first frame with images if needed
        frames = [self.env.get_obs(render=render)]
        info: dict = {}
        success = False
        cid = None

        for i, (q, g) in enumerate(path):
            action = np.zeros(14, dtype=np.float32)
            action[:7] = q.astype(np.float32)
            action[7] = g
            obs, _reward, done, info = self.env.step(action, render=render)
            frames.append(obs)

            if i == grasp_idx and cid is None:
                ee, _ = self.env.robot.get_eef_pose()
                obj, _ = self.env.object_pose()
                if np.linalg.norm(ee - obj) < 0.12:
                    cid = self.env.robot.create_grasp_constraint(self.env.object_id)
            if i == release_idx and cid is not None:
                self.env.robot.remove_constraint(cid)
                cid = None

            if done:
                success = bool(info.get("is_success", False))
                break

        if cid is not None:
            self.env.robot.remove_constraint(cid)

        if not success:
            hold = self.env.robot.get_state14()
            for _ in range(30):
                obs, _reward, done, info = self.env.step(hold, render=render)
                frames.append(obs)
                if done:
                    success = bool(info.get("is_success", False))
                    break
            success = success or self.env.is_success()

        return frames, success, {"info": info, "n_frames": len(frames)}
