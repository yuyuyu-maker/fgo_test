"""FPO Go2 observation / action helpers aligned with Isaac Lab flat training."""

from __future__ import annotations

import numpy as np

OBS_DIM = 48
ACTION_DIM = 12

# Isaac policy joint order (per-leg hip→thigh→calf). Confirm vs exported deploy.yaml.
JOINT_NAMES_ISAAC = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# Unitree SDK2 LowCmd / LowState motor index order.
JOINT_NAMES_SDK = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

DEFAULT_JOINT_POS = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=np.float32,
)

# isaac index -> sdk index
_ISAAC_TO_SDK = np.array(
    [JOINT_NAMES_SDK.index(n) for n in JOINT_NAMES_ISAAC], dtype=np.int64
)
# sdk index -> isaac index
_SDK_TO_ISAAC = np.array(
    [JOINT_NAMES_ISAAC.index(n) for n in JOINT_NAMES_SDK], dtype=np.int64
)


def isaac_to_sdk(x: np.ndarray) -> np.ndarray:
    """Reorder joint vector from Isaac policy order to SDK motor order."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros_like(x)
    out[_ISAAC_TO_SDK] = x
    return out


def sdk_to_isaac(x: np.ndarray) -> np.ndarray:
    """Reorder joint vector from SDK motor order to Isaac policy order."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros_like(x)
    out[_SDK_TO_ISAAC] = x
    return out


def projected_gravity_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction in body frame from IMU quaternion (w, x, y, z)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = q
    # R^T @ [0,0,-1] for body-frame gravity (Isaac projected_gravity)
    gx = 2.0 * (-z * x + w * y)
    gy = -2.0 * (z * y + w * x)
    gz = -1.0 + 2.0 * (x * x + y * y)
    g = np.array([gx, gy, gz], dtype=np.float32)
    n = float(np.linalg.norm(g))
    if n > 1e-8:
        g /= n
    return g


def build_obs(
    *,
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    cmd_vx: float,
    cmd_vy: float,
    cmd_yaw: float,
    joint_pos_isaac: np.ndarray,
    joint_vel_isaac: np.ndarray,
    last_action: np.ndarray,
    default_joint_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble 48-D policy observation (Isaac flat Go2 layout)."""
    q0 = DEFAULT_JOINT_POS if default_joint_pos is None else np.asarray(default_joint_pos, dtype=np.float32)
    q = np.asarray(joint_pos_isaac, dtype=np.float32).reshape(12)
    dq = np.asarray(joint_vel_isaac, dtype=np.float32).reshape(12)
    la = np.asarray(last_action, dtype=np.float32).reshape(12)
    obs = np.concatenate(
        [
            np.asarray(base_lin_vel, dtype=np.float32).reshape(3),
            np.asarray(base_ang_vel, dtype=np.float32).reshape(3),
            np.asarray(projected_gravity, dtype=np.float32).reshape(3),
            np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32),
            q - q0,
            dq,
            la,
        ],
        axis=0,
    )
    assert obs.shape == (OBS_DIM,), obs.shape
    return obs


def action_to_q_des(
    action: np.ndarray,
    *,
    action_scale: float = 0.25,
    default_joint_pos: np.ndarray | None = None,
) -> np.ndarray:
    q0 = DEFAULT_JOINT_POS if default_joint_pos is None else np.asarray(default_joint_pos, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32).reshape(12)
    return q0 + float(action_scale) * a
