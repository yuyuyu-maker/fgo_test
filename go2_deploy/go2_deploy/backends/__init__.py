"""Robot backends."""

from go2_deploy.backends.base import RobotBackend
from go2_deploy.backends.mock import MockBackend

__all__ = ["RobotBackend", "MockBackend", "make_backend"]


def make_backend(
    name: str,
    iface: str = "eth0",
    max_vx: float = 0.6,
    max_vy: float = 0.4,
    max_yaw: float = 0.8,
    **kwargs,
) -> RobotBackend:
    name = name.lower().strip()
    if name == "mock":
        return MockBackend()
    if name == "sdk2":
        from go2_deploy.backends.sdk2_sport import Sdk2SportBackend

        return Sdk2SportBackend(
            iface=iface, max_vx=max_vx, max_vy=max_vy, max_yaw=max_yaw
        )
    if name in ("fpo", "fpo_onnx"):
        from go2_deploy.backends.fpo_onnx import FpoOnnxBackend

        return FpoOnnxBackend(
            onnx_path=kwargs["onnx_path"],
            iface=iface,
            policy_hz=float(kwargs.get("policy_hz", 50.0)),
            action_scale=float(kwargs.get("action_scale", 0.25)),
            kp=float(kwargs.get("kp", 25.0)),
            kd=float(kwargs.get("kd", 0.5)),
            default_joint_pos=kwargs.get("default_joint_pos"),
            hardware=bool(kwargs.get("hardware", False)),
            max_vx=max_vx,
            max_vy=max_vy,
            max_yaw=max_yaw,
        )
    raise ValueError(f"Unknown backend '{name}'. Use mock|sdk2|fpo_onnx.")
