"""Shared velocity command state with deadman timeout."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.vx, self.vy, self.yaw


class CommandBus:
    """Thread-safe latest command; zeros out if stale."""

    def __init__(self, deadman_s: float = 0.35):
        self.deadman_s = float(deadman_s)
        self._lock = threading.Lock()
        self._cmd = VelocityCommand()
        self._stop_event = threading.Event()

    def set_velocity(self, vx: float, vy: float, yaw: float) -> None:
        with self._lock:
            self._cmd = VelocityCommand(float(vx), float(vy), float(yaw), time.monotonic())

    def stop(self) -> None:
        with self._lock:
            self._cmd = VelocityCommand(0.0, 0.0, 0.0, time.monotonic())

    def get(self) -> VelocityCommand:
        with self._lock:
            cmd = self._cmd
            age = time.monotonic() - cmd.updated_at
            if age > self.deadman_s and (cmd.vx != 0.0 or cmd.vy != 0.0 or cmd.yaw != 0.0):
                self._cmd = VelocityCommand(0.0, 0.0, 0.0, time.monotonic())
                return self._cmd
            return VelocityCommand(cmd.vx, cmd.vy, cmd.yaw, cmd.updated_at)

    def request_shutdown(self) -> None:
        self._stop_event.set()

    @property
    def shutdown(self) -> bool:
        return self._stop_event.is_set()
