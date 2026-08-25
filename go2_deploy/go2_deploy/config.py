"""Load deploy.yaml and resolve model / backend / control-mode settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "deploy.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# sport = Unitree built-in SportClient.Move
# fpo   = FPO ONNX via C++ LowCmd (web only forwards velocity cmds)
CONTROL_MODES = ("sport", "fpo")

MODE_META = {
    "sport": {
        "label": "内置步态 Sport",
        "short": "SPORT",
        "blurb": "宇树内置步态（SportClient.Move）。不是 FPO 模型。",
    },
    "fpo": {
        "label": "FPO 模型",
        "short": "FPO",
        "blurb": "自训 FPO：Python ONNX 50Hz（LowState→推理→q_des）。可选 --fpo-hardware 发 LowCmd。",
    },
}


@dataclass
class ModelSpec:
    name: str
    description: str
    fpo_variant: str
    task: str
    checkpoint: Path
    sampling_steps: int = 10
    zero_sampling: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fpo_variant": self.fpo_variant,
            "task": self.task,
            "checkpoint": str(self.checkpoint),
            "checkpoint_exists": self.checkpoint.is_file(),
            "sampling_steps": self.sampling_steps,
            "zero_sampling": self.zero_sampling,
        }


@dataclass
class PolicySpec:
    policy_hz: float = 50.0
    lowcmd_hz: float = 500.0
    action_scale: float = 0.25
    kp: float = 25.0
    kd: float = 0.5
    effort_limit: float = 23.5
    velocity_limit: float = 30.0
    default_joint_pos: list[float] = field(default_factory=list)
    pause_on_idle: bool = False
    onnx_path: Path | None = None
    aligned_yaml: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_hz": self.policy_hz,
            "lowcmd_hz": self.lowcmd_hz,
            "action_scale": self.action_scale,
            "kp": self.kp,
            "kd": self.kd,
            "effort_limit": self.effort_limit,
            "velocity_limit": self.velocity_limit,
            "default_joint_pos": list(self.default_joint_pos),
            "pause_on_idle": self.pause_on_idle,
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
            "onnx_exists": bool(self.onnx_path and self.onnx_path.is_file()),
            "aligned_yaml": str(self.aligned_yaml) if self.aligned_yaml else None,
        }


@dataclass
class DeployConfig:
    raw: dict[str, Any]
    config_path: Path
    host: str = "127.0.0.1"
    port: int = 8080
    deadman_s: float = 0.35
    control_hz: float = 50.0
    control_mode: str = "sport"
    backend_name: str = "mock"
    iface: str = "eth0"
    max_vx: float = 0.6
    max_vy: float = 0.4
    max_yaw: float = 0.8
    active_model: str = "baseline"
    models: dict[str, ModelSpec] = field(default_factory=dict)
    policy: PolicySpec = field(default_factory=PolicySpec)
    unitree_sdk: dict[str, Any] = field(default_factory=dict)
    udp_cmd_host: str | None = None
    udp_cmd_port: int = 18080

    @property
    def model(self) -> ModelSpec:
        if self.active_model not in self.models:
            raise KeyError(
                f"active_model '{self.active_model}' not in models. "
                f"Available: {sorted(self.models)}"
            )
        return self.models[self.active_model]

    @property
    def mode_meta(self) -> dict[str, str]:
        return dict(MODE_META.get(self.control_mode, MODE_META["sport"]))

    def status_payload(self) -> dict[str, Any]:
        meta = self.mode_meta
        m = self.model
        return {
            "control_mode": self.control_mode,
            "mode_label": meta["label"],
            "mode_short": meta["short"],
            "mode_blurb": meta["blurb"],
            "backend": self.backend_name,
            "control_hz": self.control_hz,
            "active_model": m.name,
            "fpo_variant": m.fpo_variant,
            "checkpoint": str(m.checkpoint),
            "checkpoint_exists": m.checkpoint.is_file(),
            "policy": self.policy.as_dict(),
            "udp_bridge": bool(self.udp_cmd_host),
            "udp_cmd": (
                f"{self.udp_cmd_host}:{self.udp_cmd_port}" if self.udp_cmd_host else None
            ),
        }


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _parse_policy(raw_policy: dict[str, Any], base_dir: Path) -> PolicySpec:
    onnx = raw_policy.get("onnx_path")
    aligned = raw_policy.get("aligned_yaml")
    dj = raw_policy.get("default_joint_pos") or []
    if not isinstance(dj, list):
        dj = []
    return PolicySpec(
        policy_hz=float(raw_policy.get("policy_hz", 50.0)),
        lowcmd_hz=float(raw_policy.get("lowcmd_hz", 500.0)),
        action_scale=float(raw_policy.get("action_scale", 0.25)),
        kp=float(raw_policy.get("kp", 25.0)),
        kd=float(raw_policy.get("kd", 0.5)),
        effort_limit=float(raw_policy.get("effort_limit", 23.5)),
        velocity_limit=float(raw_policy.get("velocity_limit", 30.0)),
        default_joint_pos=[float(x) for x in dj],
        pause_on_idle=bool(raw_policy.get("pause_on_idle", False)),
        onnx_path=_resolve_path(str(onnx), base_dir) if onnx else None,
        aligned_yaml=_resolve_path(str(aligned), base_dir) if aligned else None,
    )


def load_deploy_config(
    config_path: str | Path | None = None,
    model_override: str | None = None,
    backend_override: str | None = None,
    mode_override: str | None = None,
) -> DeployConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Deploy config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_dir = PROJECT_ROOT
    server = raw.get("server") or {}
    backend = raw.get("backend") or {}
    models_raw = raw.get("models") or {}
    policy = _parse_policy(dict(raw.get("policy") or {}), base_dir)

    models: dict[str, ModelSpec] = {}
    for name, spec in models_raw.items():
        if not isinstance(spec, dict):
            continue
        ckpt = _resolve_path(str(spec.get("checkpoint", "")), base_dir)
        models[name] = ModelSpec(
            name=name,
            description=str(spec.get("description", "")),
            fpo_variant=str(spec.get("fpo_variant", name)),
            task=str(spec.get("task", "Isaac-Velocity-Flat-Unitree-Go2-v0")),
            checkpoint=ckpt,
            sampling_steps=int(spec.get("sampling_steps", 10)),
            zero_sampling=bool(spec.get("zero_sampling", True)),
        )

    active = model_override or str(raw.get("active_model", "baseline"))
    backend_name = backend_override or str(backend.get("name", "mock"))
    control_mode = (mode_override or str(raw.get("control_mode", "sport"))).lower().strip()
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {control_mode!r}")

    return DeployConfig(
        raw=raw,
        config_path=path,
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 8080)),
        deadman_s=float(server.get("deadman_s", 0.35)),
        control_hz=float(server.get("control_hz", 50.0)),
        control_mode=control_mode,
        backend_name=backend_name,
        iface=str(backend.get("iface", "eth0")),
        max_vx=float(backend.get("max_vx", 0.6)),
        max_vy=float(backend.get("max_vy", 0.4)),
        max_yaw=float(backend.get("max_yaw", 0.8)),
        active_model=active,
        models=models,
        policy=policy,
        unitree_sdk=dict(raw.get("unitree_sdk") or {}),
    )
