"""FPO policy backend: LowState (or mock) → ONNX → q_des / LowCmd."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from go2_deploy.backends.base import RobotBackend
from go2_deploy.policy.obs import (
    DEFAULT_JOINT_POS,
    action_to_q_des,
    build_obs,
    isaac_to_sdk,
    projected_gravity_from_quat_wxyz,
    sdk_to_isaac,
)
from go2_deploy.policy.runner import OnnxPolicy, PolicyLoopState

log = logging.getLogger("go2_deploy.fpo")


class FpoOnnxBackend(RobotBackend):
    """50Hz FPO inference. Mock robot by default; optional SDK2 LowCmd.

    ``apply_velocity`` only updates the velocity-command buffer used in obs.
    The policy thread runs at ``policy_hz`` independently.
    """

    name = "fpo_onnx"

    def __init__(
        self,
        onnx_path: str,
        *,
        iface: str = "eth0",
        policy_hz: float = 50.0,
        action_scale: float = 0.25,
        kp: float = 25.0,
        kd: float = 0.5,
        default_joint_pos: list[float] | None = None,
        hardware: bool = False,
        max_vx: float = 0.6,
        max_vy: float = 0.4,
        max_yaw: float = 0.8,
    ):
        self.iface = iface
        self.policy_hz = float(policy_hz)
        self.action_scale = float(action_scale)
        self.kp = float(kp)
        self.kd = float(kd)
        self.hardware = bool(hardware)
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw = max_yaw
        self.default_joint_pos = (
            np.asarray(default_joint_pos, dtype=np.float32)
            if default_joint_pos
            else DEFAULT_JOINT_POS.copy()
        )
        self.onnx_path = onnx_path
        self._policy: OnnxPolicy | None = None
        self._state = PolicyLoopState()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._cmd_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._low_state = None
        self._low_pub = None
        self._crc = None
        self._last_log = 0.0

    def connect(self) -> None:
        self._policy = OnnxPolicy(self.onnx_path)
        if self.hardware:
            self._connect_hardware()
        self._stop.clear()
        self._thread = threading.Thread(target=self._policy_loop, name="fpo-policy", daemon=True)
        self._thread.start()
        log.info(
            "FpoOnnxBackend started policy_hz=%.1f hardware=%s onnx=%s",
            self.policy_hz,
            self.hardware,
            self.onnx_path,
        )

    def _connect_hardware(self) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, self.iface)
        self._crc = CRC()
        self._low_cmd_msg = unitree_go_msg_dds__LowCmd_()
        self._low_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._low_pub.Init()

        self._latest_lowstate = None
        self._ls_lock = threading.Lock()

        def _on_state(msg: LowState_) -> None:
            with self._ls_lock:
                self._latest_lowstate = msg

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(_on_state, 10)
        self._low_sub = sub
        log.info("FPO hardware LowState/LowCmd on iface=%s", self.iface)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("FpoOnnxBackend closed after %s ticks", self._state.ticks)

    def apply_velocity(self, vx: float, vy: float, yaw: float) -> None:
        with self._cmd_lock:
            self._cmd[0] = float(np.clip(vx, -self.max_vx, self.max_vx))
            self._cmd[1] = float(np.clip(vy, -self.max_vy, self.max_vy))
            self._cmd[2] = float(np.clip(yaw, -self.max_yaw, self.max_yaw))

    def stop(self) -> None:
        self.apply_velocity(0.0, 0.0, 0.0)

    def stand_up(self) -> None:
        log.info("stand_up ignored in FPO mode (use C++ FSM / Sport separately)")

    def stand_down(self) -> None:
        log.info("stand_down ignored in FPO mode")

    def status(self) -> dict[str, Any]:
        with self._cmd_lock:
            cmd = self._cmd.copy()
        st = self._state.as_dict()
        st["cmd"] = cmd.tolist()
        st["hardware"] = self.hardware
        st["onnx"] = self.onnx_path
        return st

    def _read_robot_state_isaac(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return lin_vel, ang_vel, gravity, q_isaac, dq_isaac."""
        if self.hardware:
            with self._ls_lock:
                msg = self._latest_lowstate
            if msg is None:
                # No state yet — stand defaults
                return (
                    np.zeros(3, np.float32),
                    np.zeros(3, np.float32),
                    np.array([0.0, 0.0, -1.0], np.float32),
                    self.default_joint_pos.copy(),
                    np.zeros(12, np.float32),
                )
            imu = msg.imu_state
            quat = np.array(imu.quaternion, dtype=np.float32)  # wxyz
            gyro = np.array(imu.gyroscope, dtype=np.float32)
            # Go2 lowstate has no direct base lin vel; use zeros (common for proprio policies
            # that were trained with lin_vel — sim2real gap; some stacks estimate from kinematics).
            lin = np.zeros(3, np.float32)
            grav = projected_gravity_from_quat_wxyz(quat)
            q_sdk = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
            dq_sdk = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)
            return lin, gyro, grav, sdk_to_isaac(q_sdk), sdk_to_isaac(dq_sdk)

        # Mock: robot at default pose, zero rates, gravity down
        return (
            np.zeros(3, np.float32),
            np.zeros(3, np.float32),
            np.array([0.0, 0.0, -1.0], np.float32),
            self.default_joint_pos.copy(),
            np.zeros(12, np.float32),
        )

    def _publish_q_des_sdk(self, q_des_isaac: np.ndarray) -> None:
        if not self.hardware or self._low_pub is None:
            return
        q_sdk = isaac_to_sdk(q_des_isaac)
        cmd = self._low_cmd_msg
        for i in range(12):
            m = cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(q_sdk[i])
            m.dq = 0.0
            m.kp = float(self.kp)
            m.kd = float(self.kd)
            m.tau = 0.0
        cmd.crc = self._crc.Crc(cmd)
        self._low_pub.Write(cmd)

    def _policy_loop(self) -> None:
        assert self._policy is not None
        dt = 1.0 / max(self.policy_hz, 1.0)
        next_t = time.monotonic()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._cmd_lock:
                vx, vy, yaw = float(self._cmd[0]), float(self._cmd[1]), float(self._cmd[2])
            lin, ang, grav, q, dq = self._read_robot_state_isaac()
            obs = build_obs(
                base_lin_vel=lin,
                base_ang_vel=ang,
                projected_gravity=grav,
                cmd_vx=vx,
                cmd_vy=vy,
                cmd_yaw=yaw,
                joint_pos_isaac=q,
                joint_vel_isaac=dq,
                last_action=self._state.last_action,
                default_joint_pos=self.default_joint_pos,
            )
            action = self._policy(obs)
            q_des = action_to_q_des(
                action,
                action_scale=self.action_scale,
                default_joint_pos=self.default_joint_pos,
            )
            self._state.last_action = action
            self._state.q_des_isaac = q_des
            self._state.ticks += 1
            self._state.last_infer_ms = (time.perf_counter() - t0) * 1000.0
            self._publish_q_des_sdk(q_des)

            now = time.monotonic()
            if now - self._last_log >= 1.0:
                log.info(
                    "fpo tick=%s infer=%.2fms cmd=(%.2f,%.2f,%.2f) action0=%.3f",
                    self._state.ticks,
                    self._state.last_infer_ms,
                    vx,
                    vy,
                    yaw,
                    float(action[0]),
                )
                self._last_log = now

            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()
