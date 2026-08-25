"""FastAPI localhost control server."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from go2_deploy.backends import RobotBackend, make_backend
from go2_deploy.command_bus import CommandBus

log = logging.getLogger("go2_deploy.server")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(
    backend_name: str = "mock",
    iface: str = "eth0",
    deadman_s: float = 0.35,
    control_hz: float = 50.0,
    deploy_cfg: Any | None = None,
    max_vx: float = 0.6,
    max_vy: float = 0.4,
    max_yaw: float = 0.8,
    udp_cmd_host: str | None = None,
    udp_cmd_port: int = 18080,
    control_mode: str = "sport",
    fpo_hardware: bool = False,
) -> FastAPI:
    bus = CommandBus(deadman_s=deadman_s)

    backend_kwargs: dict[str, Any] = {}
    if control_mode == "fpo":
        if deploy_cfg is None or deploy_cfg.policy.onnx_path is None:
            raise RuntimeError("FPO mode requires deploy_cfg.policy.onnx_path")
        onnx = deploy_cfg.policy.onnx_path
        if not onnx.is_file():
            raise FileNotFoundError(
                f"FPO ONNX missing: {onnx}\n"
                "Export first: bash scripts/export_fpo_onnx.sh --model <name>"
            )
        effective_backend = "fpo_onnx"
        pol = deploy_cfg.policy
        backend_kwargs = {
            "onnx_path": str(onnx),
            "policy_hz": pol.policy_hz,
            "action_scale": pol.action_scale,
            "kp": pol.kp,
            "kd": pol.kd,
            "default_joint_pos": list(pol.default_joint_pos) or None,
            "hardware": fpo_hardware,
        }
        if backend_name == "sdk2":
            log.warning("control_mode=fpo: ignoring Sport sdk2; using FpoOnnxBackend")
    else:
        effective_backend = backend_name
        if backend_name in ("fpo", "fpo_onnx"):
            log.warning("backend=%s with control_mode=sport is unusual; using mock", backend_name)
            effective_backend = "mock"

    backend: RobotBackend = make_backend(
        effective_backend,
        iface=iface,
        max_vx=max_vx,
        max_vy=max_vy,
        max_yaw=max_yaw,
        **backend_kwargs,
    )
    app = FastAPI(title="Go2 Deploy", version="0.2.0")
    app.state.bus = bus
    app.state.backend = backend
    app.state.control_hz = control_hz
    app.state.control_mode = control_mode
    app.state.deploy_cfg = deploy_cfg
    app.state.clients: set[WebSocket] = set()
    app.state.udp_bridge = None
    if udp_cmd_host:
        from go2_deploy.udp_cmd_bridge import UdpCommandBridge

        app.state.udp_bridge = UdpCommandBridge(host=udp_cmd_host, port=udp_cmd_port)
        log.info("UDP command bridge -> %s:%s (optional C++ mirror)", udp_cmd_host, udp_cmd_port)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def _startup() -> None:
        backend.connect()
        app.state.control_task = asyncio.create_task(_control_loop(app))
        log.info(
            "mode=%s backend=%s control_hz=%.1f deadman=%.2fs",
            control_mode,
            backend.name,
            control_hz,
            deadman_s,
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        bus.request_shutdown()
        task = getattr(app.state, "control_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        backend.stop()
        backend.close()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        cmd = bus.get()
        out: dict[str, Any] = {
            "ok": True,
            "control_mode": control_mode,
            "backend": backend.name,
            "vx": cmd.vx,
            "vy": cmd.vy,
            "yaw": cmd.yaw,
            "udp_bridge": app.state.udp_bridge is not None,
        }
        if deploy_cfg is not None:
            out.update(deploy_cfg.status_payload())
            out["backend"] = backend.name
        if hasattr(backend, "status"):
            try:
                out["policy_status"] = backend.status()  # type: ignore[attr-defined]
            except Exception as e:
                out["policy_status_error"] = str(e)
        return out

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        if deploy_cfg is None:
            return {"active_model": None, "models": [], "control_mode": control_mode}
        return {
            "control_mode": control_mode,
            "active_model": deploy_cfg.active_model,
            "models": [spec.as_dict() for spec in deploy_cfg.models.values()],
            "policy": deploy_cfg.policy.as_dict(),
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        app.state.clients.add(ws)
        try:
            hello: dict[str, Any] = {
                "type": "hello",
                "backend": backend.name,
                "control_mode": control_mode,
            }
            if deploy_cfg is not None:
                hello.update(deploy_cfg.status_payload())
                hello["backend"] = backend.name
            await ws.send_json(hello)
            while True:
                msg = await ws.receive_json()
                await _handle_client_msg(app, msg)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.warning("ws error: %s", e)
        finally:
            app.state.clients.discard(ws)
            bus.stop()

    return app


async def _handle_client_msg(app: FastAPI, msg: dict[str, Any]) -> None:
    bus: CommandBus = app.state.bus
    backend: RobotBackend = app.state.backend
    mode = getattr(app.state, "control_mode", "sport")
    mtype = msg.get("type", "cmd")

    if mtype == "stop":
        bus.stop()
        backend.stop()
        bridge = getattr(app.state, "udp_bridge", None)
        if bridge is not None:
            bridge.send(0.0, 0.0, 0.0, active=False)
        return
    if mtype == "stand_up":
        if mode != "sport":
            return
        await asyncio.to_thread(backend.stand_up)
        return
    if mtype == "stand_down":
        if mode != "sport":
            return
        await asyncio.to_thread(backend.stand_down)
        return
    if mtype == "cmd":
        active = bool(msg.get("active", True))
        if not active:
            bus.stop()
            vx = vy = yaw = 0.0
        else:
            vx = float(msg.get("vx", 0.0))
            vy = float(msg.get("vy", 0.0))
            yaw = float(msg.get("yaw", 0.0))
            bus.set_velocity(vx, vy, yaw)
        bridge = getattr(app.state, "udp_bridge", None)
        if bridge is not None:
            bridge.send(vx, vy, yaw, active=active)
        return
    log.debug("ignored msg: %s", msg)


async def _control_loop(app: FastAPI) -> None:
    """Push latest velocity command into backend at control_hz (default 50).

    Sport: SportClient.Move
    FPO: only updates command buffer; ONNX thread runs at policy_hz
    """
    bus: CommandBus = app.state.bus
    backend: RobotBackend = app.state.backend
    mode = getattr(app.state, "control_mode", "sport")
    hz = float(app.state.control_hz)
    dt = 1.0 / max(hz, 1.0)
    deploy_cfg = getattr(app.state, "deploy_cfg", None)

    while not bus.shutdown:
        cmd = bus.get()
        try:
            await asyncio.to_thread(backend.apply_velocity, cmd.vx, cmd.vy, cmd.yaw)
        except Exception as e:
            log.exception("apply_velocity failed: %s", e)

        state: dict[str, Any] = {
            "type": "state",
            "vx": cmd.vx,
            "vy": cmd.vy,
            "yaw": cmd.yaw,
            "backend": backend.name,
            "control_mode": mode,
            "ok": True,
        }
        if deploy_cfg is not None:
            state["active_model"] = deploy_cfg.active_model
            state["mode_short"] = deploy_cfg.mode_meta["short"]
        if hasattr(backend, "status"):
            try:
                ps = backend.status()  # type: ignore[attr-defined]
                state["policy_ticks"] = ps.get("ticks")
                state["infer_ms"] = ps.get("last_infer_ms")
            except Exception:
                pass

        dead: list[WebSocket] = []
        for ws in list(app.state.clients):
            try:
                await ws.send_json(state)
            except Exception:
                dead.append(ws)
        for ws in dead:
            app.state.clients.discard(ws)

        await asyncio.sleep(dt)
