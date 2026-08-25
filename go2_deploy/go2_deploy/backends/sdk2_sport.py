"""Unitree SDK2 SportClient backend (optional dependency)."""

from __future__ import annotations

import logging

from .base import RobotBackend

log = logging.getLogger("go2_deploy.sdk2")


class Sdk2SportBackend(RobotBackend):
    """High-level velocity via SportClient.Move(vx, vy, yaw)."""

    name = "sdk2"

    def __init__(self, iface: str = "eth0", max_vx: float = 0.6, max_vy: float = 0.4, max_yaw: float = 0.8):
        self.iface = iface
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw = max_yaw
        self._client = None

    def connect(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except ImportError as e:
            raise ImportError(
                "unitree_sdk2py is required for --backend sdk2.\n"
                "Install official SDK:\n"
                "  bash scripts/setup_unitree_sdk2.sh\n"
                "Repo: https://github.com/unitreerobotics/unitree_sdk2_python\n"
                "Or use --backend mock / backend.name: mock in configs/deploy.yaml."
            ) from e

        ChannelFactoryInitialize(0, self.iface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()
        self._client = client
        log.info("Sdk2SportBackend connected on iface=%s", self.iface)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.StopMove()
            except Exception:
                pass
        self._client = None
        log.info("Sdk2SportBackend closed.")

    def _clamp(self, vx: float, vy: float, yaw: float) -> tuple[float, float, float]:
        vx = max(-self.max_vx, min(self.max_vx, vx))
        vy = max(-self.max_vy, min(self.max_vy, vy))
        yaw = max(-self.max_yaw, min(self.max_yaw, yaw))
        # drop tiny noise
        if abs(vx) < 1e-3:
            vx = 0.0
        if abs(vy) < 1e-3:
            vy = 0.0
        if abs(yaw) < 1e-3:
            yaw = 0.0
        return vx, vy, yaw

    def apply_velocity(self, vx: float, vy: float, yaw: float) -> None:
        if self._client is None:
            return
        vx, vy, yaw = self._clamp(vx, vy, yaw)
        if vx == 0.0 and vy == 0.0 and yaw == 0.0:
            self._client.StopMove()
        else:
            self._client.Move(vx, vy, yaw)

    def stand_up(self) -> None:
        if self._client is None:
            return
        self._client.StandUp()
        log.info("StandUp")

    def stand_down(self) -> None:
        if self._client is None:
            return
        self._client.StandDown()
        log.info("StandDown")

    def stop(self) -> None:
        if self._client is None:
            return
        self._client.StopMove()
