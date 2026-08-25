"""Unit tests for 14D schema alignment (no PyBullet required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rm75_sim.dataset.features import build_features
from rm75_sim.schema import ROBOT_STATE_NAMES, STATE_DIM


def test_state_dim_and_names():
    assert STATE_DIM == 14
    assert len(ROBOT_STATE_NAMES) == 14
    assert ROBOT_STATE_NAMES[0] == "joint_1_rad"
    assert ROBOT_STATE_NAMES[7] == "gripper_open"
    assert ROBOT_STATE_NAMES[8] == "eef_pos_x_m"
    assert ROBOT_STATE_NAMES[-1] == "eef_rot_euler_z_rad"
    # no arm prefix for single-arm
    assert not any(n.startswith("left_") or n.startswith("right_") for n in ROBOT_STATE_NAMES)


def test_build_features_matches_recorder():
    feats = build_features({"cam0": (240, 320, 3), "cam1": (240, 320, 3)})
    assert feats["observation.state"]["shape"] == (14,)
    assert feats["action"]["shape"] == (14,)
    assert feats["observation.state"]["names"] == ROBOT_STATE_NAMES
    assert feats["action"]["names"] == ROBOT_STATE_NAMES
    assert "observation.images.cam0_rgb" in feats
    assert "observation.images.cam1_rgb" in feats
    assert "observation.images.cam0_depth" not in feats


if __name__ == "__main__":
    test_state_dim_and_names()
    test_build_features_matches_recorder()
    print("tests OK")
