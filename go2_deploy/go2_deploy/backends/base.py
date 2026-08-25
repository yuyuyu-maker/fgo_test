"""Robot backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RobotBackend(ABC):
    name: str = "base"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def apply_velocity(self, vx: float, vy: float, yaw: float) -> None: ...

    def stand_up(self) -> None:
        pass

    def stand_down(self) -> None:
        pass

    def stop(self) -> None:
        self.apply_velocity(0.0, 0.0, 0.0)
