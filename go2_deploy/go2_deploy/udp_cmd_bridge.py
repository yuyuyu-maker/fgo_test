"""Forward web velocity commands to C++ deploy via UDP (optional bridge)."""

from __future__ import annotations

import json
import logging
import socket
from typing import Any

log = logging.getLogger("go2_deploy.udp_cmd")


class UdpCommandBridge:
    """Send latest (vx,vy,yaw,active) as JSON datagrams for C++ ONNX loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18080):
        self.addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, vx: float, vy: float, yaw: float, active: bool = True) -> None:
        payload: dict[str, Any] = {
            "vx": float(vx),
            "vy": float(vy),
            "yaw": float(yaw),
            "active": bool(active),
        }
        self._sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)

    def close(self) -> None:
        self._sock.close()
