"""Write simulation episodes to LeRobot v2.1 with action[t]=state[t+1] postprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import build_features
from ..schema import DEFAULT_FPS, ROBOT_STATE_NAMES


def _postprocess_actions(states: list[np.ndarray]) -> list[np.ndarray]:
    """Same convention as dataset_recorder_node._postprocess_episode_actions."""
    actions = [s.copy() for s in states]
    for t in range(len(states) - 1):
        actions[t] = states[t + 1].copy()
    return actions


class SimLeRobotWriter:
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        task: str,
        image_hw: tuple[int, int] = (240, 320),
        camera_names: Sequence[str] = ("cam0", "cam1"),
        fps: int = DEFAULT_FPS,
        robot_type: str = "rm75_6fb",
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.task = task
        self.fps = fps
        self.camera_names = list(camera_names)
        h, w = image_hw
        shapes = {n: (h, w, 3) for n in self.camera_names}
        self.features = build_features(shapes, ROBOT_STATE_NAMES)

        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=self.features,
            root=self.root / repo_id.replace("/", "_"),
            robot_type=robot_type,
            use_videos=True,
        )

    def add_episode(self, frames: list[dict[str, Any]]) -> int:
        """frames: list of obs dicts with observation.state and image keys."""
        states = [np.asarray(f["observation.state"], dtype=np.float32).reshape(14) for f in frames]
        if any(s.shape != (14,) for s in states):
            raise ValueError("All observation.state must be shape (14,)")
        actions = _postprocess_actions(states)

        for i, f in enumerate(frames):
            frame = {
                "observation.state": states[i],
                "action": actions[i],
            }
            for cam in self.camera_names:
                key = f"observation.images.{cam}_rgb"
                img = np.asarray(f[key], dtype=np.uint8)
                if img.ndim != 3 or img.shape[2] != 3:
                    raise ValueError(f"{key} must be HxWx3 uint8, got {img.shape}")
                frame[key] = img
            self.dataset.add_frame(frame, task=self.task)

        self.dataset.save_episode()
        return len(frames)

    @property
    def dataset_path(self) -> Path:
        return Path(self.dataset.root)
