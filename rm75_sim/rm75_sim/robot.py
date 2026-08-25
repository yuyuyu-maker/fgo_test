"""PyBullet RM75-6FB single-arm loader: home, joints, FK, IK, gripper, RGB cams."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

from .schema import HOME_JOINT_DEG, JOINT_NAMES

ASSETS = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_URDF = ASSETS / "RM75-6FB_pybullet.urdf"


class RM75Robot:
    """Fixed-base RM75 with synthetic parallel gripper (joint name ``gripper``)."""

    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF,
        gui: bool = False,
        timestep: float = 1.0 / 240.0,
    ):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)

        self.gui = gui
        self.timestep = timestep
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(str(ASSETS))
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(timestep)

        self.plane = p.loadURDF("plane.urdf")
        # URDF package-relative meshes resolve from cwd of urdf parent
        self.robot = p.loadURDF(
            str(self.urdf_path),
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )

        self._arm_joint_indices: list[int] = []
        self._gripper_index: Optional[int] = None
        self._gripper_mimic_index: Optional[int] = None
        self._ee_link_index = -1
        self._name_to_idx: dict[str, int] = {}

        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            jname = info[1].decode("utf-8")
            jtype = info[2]
            link_name = info[12].decode("utf-8")
            self._name_to_idx[jname] = i
            if jname in JOINT_NAMES[:7] and jtype == p.JOINT_REVOLUTE:
                self._arm_joint_indices.append(i)
            if jname == "gripper" and jtype == p.JOINT_PRISMATIC:
                self._gripper_index = i
            if jname == "gripper_mimic" and jtype == p.JOINT_PRISMATIC:
                self._gripper_mimic_index = i
            if link_name == "ee_tip":
                self._ee_link_index = i

        if len(self._arm_joint_indices) != 7:
            raise RuntimeError(
                f"Expected 7 arm joints, found {self._arm_joint_indices} "
                f"(names={list(self._name_to_idx)})"
            )
        if self._gripper_index is None:
            raise RuntimeError("gripper prismatic joint not found in URDF")
        if self._ee_link_index < 0:
            # fall back to last arm link (link_7)
            self._ee_link_index = self._arm_joint_indices[-1]

        self.home_joint_rad = np.deg2rad(np.asarray(HOME_JOINT_DEG, dtype=np.float64))
        # gripper prismatic: 0=closed, 0.03=open → gripper_open 0..1
        self._grip_travel = 0.03

    def disconnect(self) -> None:
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

    def joint_order_table(self) -> list[dict]:
        rows = []
        for k, name in enumerate(JOINT_NAMES[:7]):
            rows.append(
                {
                    "schema_name": name,
                    "state_name": f"{name}_rad",
                    "pybullet_index": self._arm_joint_indices[k],
                    "unit": "rad",
                }
            )
        rows.append(
            {
                "schema_name": "gripper",
                "state_name": "gripper_open",
                "pybullet_index": self._gripper_index,
                "unit": "normalized_open_0_1",
                "note": "maps prismatic travel / 0.03",
            }
        )
        return rows

    def reset_home(self, gripper_open: float = 1.0) -> None:
        for idx, q in zip(self._arm_joint_indices, self.home_joint_rad):
            p.resetJointState(self.robot, idx, float(q))
        self.set_gripper_open(gripper_open, reset=True)
        for _ in range(10):
            p.stepSimulation()

    def set_arm_joints(self, q7: np.ndarray, reset: bool = False) -> None:
        q7 = np.asarray(q7, dtype=np.float64).reshape(7)
        for idx, q in zip(self._arm_joint_indices, q7):
            if reset:
                p.resetJointState(self.robot, idx, float(q))
            else:
                p.setJointMotorControl2(
                    self.robot,
                    idx,
                    p.POSITION_CONTROL,
                    targetPosition=float(q),
                    force=80.0,
                    maxVelocity=2.0,
                )

    def set_gripper_open(self, open01: float, reset: bool = False) -> None:
        open01 = float(np.clip(open01, 0.0, 1.0))
        travel = open01 * self._grip_travel
        for j in (self._gripper_index, self._gripper_mimic_index):
            if j is None:
                continue
            if reset:
                p.resetJointState(self.robot, j, travel)
            else:
                p.setJointMotorControl2(
                    self.robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=travel,
                    force=80.0,
                    maxVelocity=0.5,
                )

    def get_arm_joints(self) -> np.ndarray:
        return np.array(
            [p.getJointState(self.robot, i)[0] for i in self._arm_joint_indices],
            dtype=np.float64,
        )

    def get_gripper_open(self) -> float:
        pos = p.getJointState(self.robot, self._gripper_index)[0]
        return float(np.clip(pos / self._grip_travel, 0.0, 1.0))

    def get_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_xyz_m, euler_xyz_rad) of ee_tip in world frame."""
        state = p.getLinkState(self.robot, self._ee_link_index, computeForwardKinematics=True)
        pos = np.array(state[4], dtype=np.float64)
        orn = state[5]
        euler = np.array(p.getEulerFromQuaternion(orn), dtype=np.float64)
        return pos, euler

    def get_state14(self) -> np.ndarray:
        q = self.get_arm_joints()
        g = self.get_gripper_open()
        pos, euler = self.get_eef_pose()
        return np.concatenate([q, [g], pos, euler]).astype(np.float32)

    def ik(
        self,
        target_pos: np.ndarray,
        target_euler: Optional[np.ndarray] = None,
        max_iters: int = 200,
        n_restarts: int = 4,
        position_only: bool = True,
    ) -> np.ndarray:
        """Inverse kinematics to ``target_pos``.

        Default ``position_only=True``: orientation is often infeasible with our
        synthetic gripper tip and breaks reachability; script experts use pos-only.
        """
        orn = None
        if target_euler is not None and not position_only:
            orn = p.getQuaternionFromEuler(list(target_euler))

        lower, upper, ranges = [], [], []
        for idx in self._arm_joint_indices:
            info = p.getJointInfo(self.robot, idx)
            lo, hi = float(info[8]), float(info[9])
            if hi < lo + 1e-6:
                lo, hi = -np.pi, np.pi
            lower.append(lo)
            upper.append(hi)
            ranges.append(hi - lo)
        n_move = len(
            [
                i
                for i in range(p.getNumJoints(self.robot))
                if p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED
            ]
        )
        while len(lower) < n_move:
            lower.append(-1.0)
            upper.append(1.0)
            ranges.append(2.0)

        stretch = np.deg2rad([0.0, 25.0, 0.0, 70.0, 0.0, 70.0, 0.0])
        seeds = [self.get_arm_joints(), stretch]
        rng = np.random.default_rng(0)
        for _ in range(max(0, n_restarts - 2)):
            seeds.append(np.array([rng.uniform(lower[i], upper[i]) for i in range(7)]))

        best_q = seeds[0].copy()
        best_err = 1e9
        for seed in seeds:
            for i, idx in enumerate(self._arm_joint_indices):
                p.resetJointState(self.robot, idx, float(seed[i]))
            rest_full = list(seed) + [0.0] * (n_move - 7)
            kwargs = dict(
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=ranges,
                restPoses=rest_full,
                maxNumIterations=max_iters,
                residualThreshold=1e-5,
            )
            if orn is None:
                q = p.calculateInverseKinematics(
                    self.robot, self._ee_link_index, list(target_pos), **kwargs
                )
            else:
                q = p.calculateInverseKinematics(
                    self.robot, self._ee_link_index, list(target_pos), orn, **kwargs
                )
            q7 = np.array(q[:7], dtype=np.float64)
            for i, idx in enumerate(self._arm_joint_indices):
                p.resetJointState(self.robot, idx, float(q7[i]))
            pos, _ = self.get_eef_pose()
            err = float(np.linalg.norm(pos - target_pos))
            if err < best_err:
                best_err = err
                best_q = q7
        for i, idx in enumerate(self._arm_joint_indices):
            p.resetJointState(self.robot, idx, float(best_q[i]))
        return best_q

    def create_grasp_constraint(self, body_id: int) -> int:
        """Attach object to ee_tip (script-expert soft grasp)."""
        ee_pos, _ = self.get_eef_pose()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(body_id)
        inv_ee = p.invertTransform(ee_pos, p.getLinkState(self.robot, self._ee_link_index)[5])
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee[0], inv_ee[1], obj_pos, obj_orn)
        cid = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self._ee_link_index,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        return cid

    def remove_constraint(self, cid: Optional[int]) -> None:
        if cid is not None:
            p.removeConstraint(cid)

    def step_control(
        self,
        q7: np.ndarray,
        gripper_open: float,
        n_substeps: int = 8,
    ) -> None:
        self.set_arm_joints(q7, reset=False)
        self.set_gripper_open(gripper_open, reset=False)
        for _ in range(n_substeps):
            p.stepSimulation()

    def render_rgb(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        width: int = 320,
        height: int = 240,
        fov: float = 60.0,
        near: float = 0.01,
        far: float = 2.5,
    ) -> np.ndarray:
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov, width / height, near, far)
        _, _, rgb, _, _ = p.getCameraImage(
            width,
            height,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.asarray(rgb, dtype=np.uint8)[:, :, :3]
        return rgb
