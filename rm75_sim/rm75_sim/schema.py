"""Single-arm 14D schema aligned with realman_teleop_v2 utils / recorder.

Source of truth (do not rename silently):
  realman_teleop_v2/.../realman_teleop/utils.py :: ROBOT_STATE_NAMES
  dataset_recorder_node.py :: build_features()
"""

from __future__ import annotations

# Matches teleop utils.ROBOT_JOINT_NAMES / EXO_JOINT_NAMES
JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
]

# Matches teleop utils.ROBOT_STATE_NAMES (14-D, single arm, no left/right prefix)
ROBOT_STATE_NAMES = [
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "joint_7_rad",
    "gripper_open",
    "eef_pos_x_m",
    "eef_pos_y_m",
    "eef_pos_z_m",
    "eef_rot_euler_x_rad",
    "eef_rot_euler_y_rad",
    "eef_rot_euler_z_rad",
]

STATE_DIM = len(ROBOT_STATE_NAMES)  # 14

# From teleop_params.yaml realman_driver_node.home_joint_deg (degrees)
HOME_JOINT_DEG = [-0.263, -0.505, -0.157, 43.953, 1.225, 121.751, -0.187]

# Default cameras matching teleop_params camera_names
DEFAULT_CAMERA_NAMES = ["cam0", "cam1"]

# Record rate matching dataset_recorder_node record_hz
DEFAULT_FPS = 30
