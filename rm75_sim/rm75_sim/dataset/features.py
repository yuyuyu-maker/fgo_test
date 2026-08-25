"""LeRobot feature schema matching realman dataset_recorder_node.build_features."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..schema import ROBOT_STATE_NAMES


def build_features(
    camera_shapes: Dict[str, Tuple[int, int, int]],
    state_names: List[str] | None = None,
) -> dict:
    """Mirror realman_teleop build_features (RGB only, use_depth=False).

    camera_shapes: cam_name -> (H, W, C)
    """
    state_names = list(state_names) if state_names else list(ROBOT_STATE_NAMES)
    state_shape = (len(state_names),)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
    }
    for cam_name, shape in camera_shapes.items():
        height, width, channels = shape
        features[f"observation.images.{cam_name}_rgb"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
        }
    return features
