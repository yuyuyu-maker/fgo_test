"""Mock backend: log commands, no hardware."""

from __future__ import annotations

import logging
import time

from .base import RobotBackend

log = logging.getLogger("go2_deploy.mock")


class MockBackend(RobotBackend):
    name = "mock"

    def __init__(self, log_every_s: float = 0.5):
        self.log_every_s = log_every_s
        self._last_log = 0.0
        self._last = (0.0, 0.0, 0.0)

    def connect(self) -> None:
        log.info("MockBackend connected (no robot).")

    def close(self) -> None:
        log.info("MockBackend closed.")

    def apply_velocity(self, vx: float, vy: float, yaw: float) -> None:
        self._last = (vx, vy, yaw)
        now = time.monotonic()
        moving = abs(vx) + abs(vy) + abs(yaw) > 1e-4
        if moving and now - self._last_log >= self.log_every_s:
            log.info("cmd vx=%.3f vy=%.3f yaw=%.3f", vx, vy, yaw)
            self._last_log = now

    def stand_up(self) -> None:
        log.info("stand_up (mock)")

    def stand_down(self) -> None:
        log.info("stand_down (mock)")
