"""Square-style insertion: block into a narrow slot (strict geometry, not a wide dish)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pybullet as p

from ..robot import RM75Robot
from ..schema import DEFAULT_CAMERA_NAMES, ROBOT_STATE_NAMES


@dataclass
class SquareConfig:
    table_height: float = 0.40
    table_size: tuple[float, float, float] = (0.8, 0.6, 0.04)
    # nut / square block
    block_size: tuple[float, float, float] = (0.04, 0.04, 0.04)
    # narrow slot: clearance ~2mm each side
    slot_inner: tuple[float, float, float] = (0.044, 0.044, 0.05)
    slot_wall: float = 0.012
    block_spawn_center: tuple[float, float] = (0.30, 0.10)
    block_xy_range: tuple[float, float] = (0.04, 0.05)
    slot_center: tuple[float, float] = (0.32, -0.08)
    max_steps: int = 500
    success_hold_steps: int = 20
    # success: block XY inside slot, yaw aligned, and inserted (z below threshold)
    insert_z_max: float = 0.42  # table 0.40 + small clearance once seated
    yaw_tol_rad: float = 0.15
    xy_tol: float = 0.012
    image_size: tuple[int, int] = (240, 320)
    camera_names: tuple[str, ...] = tuple(DEFAULT_CAMERA_NAMES)


class SquareEnv:
    def __init__(self, gui: bool = False, cfg: Optional[SquareConfig] = None, seed: int = 0):
        self.cfg = cfg or SquareConfig()
        self.rng = np.random.default_rng(seed)
        self.robot = RM75Robot(gui=gui)
        self.table_id: Optional[int] = None
        self.block_id: Optional[int] = None
        self.slot_ids: list[int] = []
        self._step_count = 0
        self._success_streak = 0
        self._build_scene()

    def close(self) -> None:
        self.robot.disconnect()

    def _build_scene(self) -> None:
        c = self.cfg
        half = [s / 2 for s in c.table_size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=[0.5, 0.45, 0.4, 1]
        )
        self.table_id = p.createMultiBody(
            0, col, vis, basePosition=[0.4, 0.0, c.table_height - half[2]]
        )

        sx, sy = c.slot_center
        ix, iy, iz = c.slot_inner
        t = c.slot_wall
        z0 = c.table_height
        # Four walls forming a square pocket (inner opening = slot_inner)
        walls = [
            ([sx, sy + (iy / 2 + t / 2), z0 + iz / 2], [ix / 2 + t, t / 2, iz / 2]),
            ([sx, sy - (iy / 2 + t / 2), z0 + iz / 2], [ix / 2 + t, t / 2, iz / 2]),
            ([sx + (ix / 2 + t / 2), sy, z0 + iz / 2], [t / 2, iy / 2, iz / 2]),
            ([sx - (ix / 2 + t / 2), sy, z0 + iz / 2], [t / 2, iy / 2, iz / 2]),
        ]
        self.slot_ids = []
        for pos, he in walls:
            col_w = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
            vis_w = p.createVisualShape(
                p.GEOM_BOX, halfExtents=he, rgbaColor=[0.15, 0.15, 0.7, 1]
            )
            self.slot_ids.append(p.createMultiBody(0, col_w, vis_w, basePosition=pos))

    def _spawn_block(self) -> None:
        c = self.cfg
        if self.block_id is not None:
            p.removeBody(self.block_id)
        rx, ry = c.block_xy_range
        ox = c.block_spawn_center[0] + self.rng.uniform(-rx, rx)
        oy = c.block_spawn_center[1] + self.rng.uniform(-ry, ry)
        yaw = self.rng.uniform(-0.25, 0.25)
        he = [s / 2 for s in c.block_size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.95, 0.75, 0.1, 1])
        z = c.table_height + he[2] + 0.001
        self.block_id = p.createMultiBody(
            0.2,
            col,
            vis,
            basePosition=[ox, oy, z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        )
        p.changeDynamics(self.block_id, -1, lateralFriction=1.4)

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.robot.reset_home(gripper_open=1.0)
        self._spawn_block()
        for _ in range(30):
            p.stepSimulation()
        self._step_count = 0
        self._success_streak = 0
        return self.get_obs()

    def block_pose(self) -> tuple[np.ndarray, float]:
        pos, orn = p.getBasePositionAndOrientation(self.block_id)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(pos), float(euler[2])

    def is_success(self) -> bool:
        c = self.cfg
        pos, yaw = self.block_pose()
        sx, sy = c.slot_center
        xy_ok = abs(pos[0] - sx) < c.xy_tol and abs(pos[1] - sy) < c.xy_tol
        # wrap yaw to [-pi, pi] and compare to 0 (slot axis-aligned)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_ok = abs(yaw) < c.yaw_tol_rad or abs(abs(yaw) - np.pi / 2) < c.yaw_tol_rad
        z_ok = pos[2] <= c.insert_z_max
        return bool(xy_ok and yaw_ok and z_ok)

    def get_images(self) -> dict[str, np.ndarray]:
        c = self.cfg
        h, w = c.image_size
        ee, _ = self.robot.get_eef_pose()
        cam0 = self.robot.render_rgb(
            eye=ee + np.array([-0.05, 0.0, 0.12]),
            target=ee + np.array([0.15, 0.0, -0.05]),
            width=w,
            height=h,
        )
        cam1 = self.robot.render_rgb(
            eye=np.array([0.95, 0.5, 1.0]),
            target=np.array([0.45, 0.0, 0.42]),
            width=w,
            height=h,
        )
        return {"cam0": cam0, "cam1": cam1}

    def get_obs(self) -> dict[str, Any]:
        state = self.robot.get_state14()
        images = self.get_images()
        obs = {"observation.state": state, "state_names": list(ROBOT_STATE_NAMES)}
        for name in self.cfg.camera_names:
            obs[f"observation.images.{name}_rgb"] = images[name]
        return obs

    def step(self, action14: np.ndarray) -> tuple[dict[str, Any], float, bool, dict]:
        a = np.asarray(action14, dtype=np.float32).reshape(14)
        self.robot.step_control(a[:7], float(a[7]), n_substeps=8)
        self._step_count += 1
        success_now = self.is_success()
        self._success_streak = self._success_streak + 1 if success_now else 0
        terminated = self._success_streak >= self.cfg.success_hold_steps
        truncated = self._step_count >= self.cfg.max_steps
        done = terminated or truncated
        reward = 1.0 if terminated else 0.0
        info = {"is_success": terminated, "success_now": success_now, "steps": self._step_count}
        return self.get_obs(), reward, done, info
