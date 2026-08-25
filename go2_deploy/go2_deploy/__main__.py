"""CLI: python -m go2_deploy"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from go2_deploy.config import CONTROL_MODES, DEFAULT_CONFIG, load_deploy_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 localhost deploy console")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override active_model in YAML (e.g. baseline, reflow, reward_aware)",
    )
    parser.add_argument(
        "--mode",
        choices=CONTROL_MODES,
        default=None,
        help="sport=内置步态 SportClient; fpo=FPO ONNX 推理",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print models from YAML and exit",
    )
    parser.add_argument("--host", default=None, help="Override server.host")
    parser.add_argument("--port", type=int, default=None, help="Override server.port")
    parser.add_argument("--backend", choices=("mock", "sdk2"), default=None)
    parser.add_argument("--iface", default=None, help="Override backend.iface")
    parser.add_argument("--deadman", type=float, default=None)
    parser.add_argument("--hz", type=float, default=None, help="Web/cmd push rate (default 50)")
    parser.add_argument(
        "--udp-cmd",
        default=None,
        help="Also forward cmds to host:port (optional C++ mirror)",
    )
    parser.add_argument(
        "--fpo-hardware",
        action="store_true",
        help="FPO mode: publish LowCmd / read LowState via SDK2 (dangerous)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("go2_deploy")

    cfg = load_deploy_config(
        config_path=args.config,
        model_override=args.model,
        backend_override=args.backend,
        mode_override=args.mode,
    )
    if args.iface is not None:
        cfg.iface = args.iface
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.deadman is not None:
        cfg.deadman_s = args.deadman
    if args.hz is not None:
        cfg.control_hz = args.hz

    # Keep teleop stream aligned with policy by default
    if cfg.control_mode == "fpo" and args.hz is None:
        cfg.control_hz = float(cfg.policy.policy_hz)

    udp_host, udp_port = None, 18080
    if args.udp_cmd:
        host_port = args.udp_cmd.rsplit(":", 1)
        udp_host = host_port[0]
        udp_port = int(host_port[1]) if len(host_port) == 2 else 18080

    cfg.udp_cmd_host = udp_host
    cfg.udp_cmd_port = udp_port

    if args.list_models:
        print(f"config: {cfg.config_path}")
        print(f"control_mode: {cfg.control_mode} ({cfg.mode_meta['label']})")
        print(f"active_model: {cfg.active_model}")
        pol = cfg.policy
        print(
            f"policy: hz={pol.policy_hz} scale={pol.action_scale} kp={pol.kp} kd={pol.kd}"
        )
        print(f"onnx: {pol.onnx_path} exists={pol.onnx_path.is_file() if pol.onnx_path else False}")
        for name, spec in sorted(cfg.models.items()):
            mark = "*" if name == cfg.active_model else " "
            exists = "OK" if spec.checkpoint.is_file() else "MISSING"
            print(f" {mark} {name:22s} [{exists}] {spec.checkpoint}")
            if spec.description:
                print(f"   {spec.description}")
        return

    model = cfg.model
    log.info("config=%s", cfg.config_path)
    log.info("control_mode=%s (%s)", cfg.control_mode, cfg.mode_meta["label"])
    log.info(
        "model=%s variant=%s ckpt=%s exists=%s",
        model.name,
        model.fpo_variant,
        model.checkpoint,
        model.checkpoint.is_file(),
    )
    log.info(
        "policy aligned: hz=%.1f scale=%.2f kp=%.1f kd=%.1f onnx=%s exists=%s",
        cfg.policy.policy_hz,
        cfg.policy.action_scale,
        cfg.policy.kp,
        cfg.policy.kd,
        cfg.policy.onnx_path,
        cfg.policy.onnx_path.is_file() if cfg.policy.onnx_path else False,
    )

    if cfg.control_mode == "fpo":
        from go2_deploy.config import PROJECT_ROOT

        model_onnx = PROJECT_ROOT / "exported" / cfg.active_model / "policy.onnx"
        if model_onnx.is_file():
            cfg.policy.onnx_path = model_onnx
        if cfg.policy.onnx_path is None or not cfg.policy.onnx_path.is_file():
            raise SystemExit(
                f"FPO mode needs ONNX at exported/{cfg.active_model}/policy.onnx "
                f"(or policy.onnx_path). Run: bash scripts/export_fpo_onnx.sh --model {cfg.active_model}"
            )
        log.info("FPO ONNX=%s hardware=%s", cfg.policy.onnx_path, args.fpo_hardware)
    sdk = cfg.unitree_sdk
    if cfg.control_mode == "sport" and cfg.backend_name == "sdk2":
        log.info(
            "backend=sdk2 iface=%s sdk_github=%s",
            cfg.iface,
            sdk.get("github", "https://github.com/unitreerobotics/unitree_sdk2_python"),
        )

    from go2_deploy.server import create_app

    app = create_app(
        backend_name=cfg.backend_name,
        iface=cfg.iface,
        deadman_s=cfg.deadman_s,
        control_hz=cfg.control_hz,
        deploy_cfg=cfg,
        max_vx=cfg.max_vx,
        max_vy=cfg.max_vy,
        max_yaw=cfg.max_yaw,
        udp_cmd_host=udp_host,
        udp_cmd_port=udp_port,
        control_mode=cfg.control_mode,
        fpo_hardware=bool(args.fpo_hardware),
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
