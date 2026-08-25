"""ONNX policy inference for FPO Go2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from go2_deploy.policy.obs import ACTION_DIM, OBS_DIM

log = logging.getLogger("go2_deploy.policy")


class OnnxPolicy:
    """Run exported FPO policy.onnx (obs → actions; normalizer baked into graph)."""

    def __init__(self, onnx_path: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime required for FPO inference. pip install onnxruntime"
            ) from e

        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX not found: {path}")
        self.path = path
        self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self._sess.get_inputs()
        outputs = self._sess.get_outputs()
        self._in_name = inputs[0].name
        self._out_name = outputs[0].name
        log.info(
            "OnnxPolicy loaded %s in=%s out=%s",
            path,
            getattr(inputs[0], "shape", None),
            getattr(outputs[0], "shape", None),
        )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM)
        out = self._sess.run([self._out_name], {self._in_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(ACTION_DIM)


class PolicyLoopState:
    """Mutable buffers for the policy loop."""

    def __init__(self) -> None:
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.q_des_isaac = np.zeros(ACTION_DIM, dtype=np.float32)
        self.ticks = 0
        self.last_infer_ms = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "last_infer_ms": self.last_infer_ms,
            "last_action": self.last_action.tolist(),
            "q_des": self.q_des_isaac.tolist(),
        }
