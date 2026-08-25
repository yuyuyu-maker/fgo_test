#!/usr/bin/env python3
"""M0 smoke: load RM75, print joint map, go home, open/close gripper, tiny IK."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rm75_sim.robot import RM75Robot
from rm75_sim.schema import HOME_JOINT_DEG, ROBOT_STATE_NAMES


def main() -> int:
    robot = RM75Robot(gui=False)
    table = robot.joint_order_table()
    out = ROOT / "assets" / "joint_map.json"
    out.write_text(json.dumps(table, indent=2))
    print("Joint map written to", out)
    for row in table:
        print(row)

    robot.reset_home(gripper_open=1.0)
    s0 = robot.get_state14()
    assert s0.shape == (14,), s0.shape
    print("HOME_JOINT_DEG:", HOME_JOINT_DEG)
    print("state@home names:", ROBOT_STATE_NAMES)
    print("joints_rad:", s0[:7])
    print("gripper_open:", float(s0[7]))
    print("eef_pos_m:", s0[8:11])
    print("eef_euler_rad:", s0[11:14])

    robot.set_gripper_open(0.0)
    for _ in range(50):
        p.stepSimulation()
    print("gripper after close:", robot.get_gripper_open())
    robot.set_gripper_open(1.0)
    for _ in range(50):
        p.stepSimulation()
    print("gripper after open:", robot.get_gripper_open())

    pos, eul = robot.get_eef_pose()
    q = robot.ik(pos + np.array([0.05, 0.0, -0.05]), eul)
    robot.step_control(q, 1.0, n_substeps=40)
    pos2, _ = robot.get_eef_pose()
    print("IK delta pos:", pos2 - pos)

    robot.disconnect()
    print("M0 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
