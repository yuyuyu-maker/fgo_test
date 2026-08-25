"""Can-style pick-and-place: cylinder into circular bin on a table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pybullet as p

from ..robot import RM75Robot
from ..schema import DEFAULT_CAMERA_NAMES, ROBOT_STATE_NAMES


@dataclass
class CanConfig:
    table_height: float = 0.40
    table_size: tuple[float, float, float] = (0.8, 0.6, 0.04)
    object_radius: float = 0.025
    object_height: float = 0.08
    bin_inner_radius: float = 0.06
    bin_outer_radius: float = 0.075
    bin_height: float = 0.04
    # object spawn region on table (relative to table center)
    object_xy_range: tuple[float, float] = (0.04, 0.06)  # half-range around spawn center
    object_spawn_center: tuple[float, float] = (0.32, 0.08)
    bin_center: tuple[float, float] = (0.32, -0.10)
    max_steps: int = 400
    success_hold_steps: int = 15
    image_size: tuple[int, int] = (240, 320)  # H, W
    camera_names: tuple[str, ...] = tuple(DEFAULT_CAMERA_NAMES)


class CanEnv:
    """Minimal Can env. Observation keys match realman LeRobot schema."""

    def __init__(self, gui: bool = False, cfg: Optional[CanConfig] = None, seed: int = 0):
        self.cfg = cfg or CanConfig()
        self.rng = np.random.default_rng(seed)
        self.robot = RM75Robot(gui=gui)
        self.table_id: Optional[int] = None
        self.object_id: Optional[int] = None
        self.bin_ids: list[int] = []
        self._step_count = 0
        self._success_streak = 0
        self._done = False
        self._build_scene()

    def close(self) -> None:
        self.robot.disconnect()

    def _build_scene(self) -> None:
        c = self.cfg
        half = [s / 2 for s in c.table_size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.4, 0.25, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.4, 0.0, c.table_height - half[2]],
        )

        # annular bin approx: outer cylinder (static) + inner hole via thin walls as 4 boxes
        bx, by = c.bin_center
        z0 = c.table_height
        self.bin_ids = []
        wall_t = c.bin_outer_radius - c.bin_inner_radius
        for ang in (0, 90, 180, 270):
            rad = np.deg2rad(ang)
            cx = bx + (c.bin_inner_radius + wall_t / 2) * np.cos(rad)
            cy = by + (c.bin_inner_radius + wall_t / 2) * np.sin(rad)
            # tangential wall segment
            hx = wall_t / 2 + 0.005
            hy = c.bin_inner_radius * 0.9
            if ang % 180 == 0:
                half_ext = [hx, hy, c.bin_height / 2]
            else:
                half_ext = [hy, hx, c.bin_height / 2]
            col_w = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
            vis_w = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_ext, rgbaColor=[0.1, 0.6, 0.2, 1]
            )
            bid = p.createMultiBody(
                0,
                col_w,
                vis_w,
                basePosition=[cx, cy, z0 + c.bin_height / 2],
            )
            self.bin_ids.append(bid)

        # floor disk visual for target region
        vis_disk = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=c.bin_inner_radius,
            length=0.002,
            rgbaColor=[0.2, 0.85, 0.3, 0.5],
        )
        self.bin_ids.append(
            p.createMultiBody(
                0,
                -1,
                vis_disk,
                basePosition=[bx, by, z0 + 0.001],
            )
        )

    def _spawn_object(self) -> None:
        c = self.cfg
        if self.object_id is not None:
            p.removeBody(self.object_id)
        rx, ry = c.object_xy_range
        ox = c.object_spawn_center[0] + self.rng.uniform(-rx, rx)
        oy = c.object_spawn_center[1] + self.rng.uniform(-ry, ry)
        yaw = self.rng.uniform(-0.3, 0.3)
        col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=c.object_radius, height=c.object_height
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=c.object_radius,
            length=c.object_height,
            rgbaColor=[0.9, 0.2, 0.15, 1],
        )
        z = c.table_height + c.object_height / 2 + 0.001
        self.object_id = p.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[ox, oy, z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        )
        p.changeDynamics(self.object_id, -1, lateralFriction=1.2, rollingFriction=0.01)

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.robot.reset_home(gripper_open=1.0)
        self._spawn_object()
        for _ in range(30):
            p.stepSimulation()
        self._step_count = 0
        self._success_streak = 0
        self._done = False
        return self.get_obs()

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos), np.array(orn)

    def is_success(self) -> bool:
        """Object XY inside bin inner radius and resting near table+bin floor."""
        c = self.cfg
        pos, _ = self.object_pose()
        bx, by = c.bin_center
        dist = float(np.hypot(pos[0] - bx, pos[1] - by))
        z_ok = c.table_height - 0.01 <= pos[2] <= c.table_height + c.bin_height + 0.03
        return dist <= c.bin_inner_radius * 0.85 and z_ok

    def get_images(self) -> dict[str, np.ndarray]:
        c = self.cfg
        h, w = c.image_size
        ee, _ = self.robot.get_eef_pose()
        # cam0: wrist-ish (slightly behind/above EE looking at object area)
        cam0 = self.robot.render_rgb(
            eye=ee + np.array([-0.05, 0.0, 0.12]),
            target=ee + np.array([0.15, 0.0, -0.05]),
            width=w,
            height=h,
        )
        # cam1: fixed third-person
        cam1 = self.robot.render_rgb(
            eye=np.array([0.9, 0.55, 0.95]),
            target=np.array([0.45, 0.0, 0.42]),
            width=w,
            height=h,
        )
        return {"cam0": cam0, "cam1": cam1}

    def get_obs(self, render: bool = True) -> dict[str, Any]:
        state = self.robot.get_state14()
        obs = {
            "observation.state": state,
            "state_names": list(ROBOT_STATE_NAMES),
        }
        if render:
            images = self.get_images()
            for name in self.cfg.camera_names:
                obs[f"observation.images.{name}_rgb"] = images[name]
        else:
            h, w = self.cfg.image_size
            for name in self.cfg.camera_names:
                obs[f"observation.images.{name}_rgb"] = np.zeros((h, w, 3), dtype=np.uint8)
        return obs

    def step(
        self, action14: np.ndarray, render: bool = True
    ) -> tuple[dict[str, Any], float, bool, dict]:
        """Action is next desired 14D state (same schema as observation.state)."""
        a = np.asarray(action14, dtype=np.float32).reshape(14)
        q7 = a[:7]
        g = float(a[7])
        self.robot.step_control(q7, g, n_substeps=8)
        self._step_count += 1

        success_now = self.is_success()
        if success_now:
            self._success_streak += 1
        else:
            self._success_streak = 0

        terminated = self._success_streak >= self.cfg.success_hold_steps
        truncated = self._step_count >= self.cfg.max_steps
        done = terminated or truncated
        reward = 1.0 if terminated else 0.0
        info = {
            "is_success": terminated,
            "success_now": success_now,
            "steps": self._step_count,
        }
        return self.get_obs(render=render), reward, done, info
