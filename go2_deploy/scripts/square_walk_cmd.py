#!/usr/bin/env python3
"""Inject square-velocity commands into a running go2_deploy FPO server (WebSocket).

Same schedule as Isaac square walk: forward / left / back / right at |v|=SPEED for SEGMENT_S each.

Usage (terminal A):
  python -m go2_deploy --mode fpo --model ours_s64 --host 127.0.0.1
Usage (terminal B):
  python scripts/square_walk_cmd.py --speed 0.3 --segment-s 3 --cycles 3 --log out/cmd_hw.csv

Requires the deploy server WebSocket at ws://host:port/ws (see go2_deploy/server.py).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

DIRECTIONS = (
    ("forward", 1.0, 0.0, 0.0),
    ("left", 0.0, 1.0, 0.0),
    ("backward", -1.0, 0.0, 0.0),
    ("right", 0.0, -1.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--segment-s", type=float, default=3.0)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--warmup-s", type=float, default=1.0)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--log", type=Path, default=Path("square_walk_cmd_log.csv"))
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    try:
        import websockets
    except ImportError as e:
        raise SystemExit("pip install websockets") from e

    dt = 1.0 / args.hz
    total_s = args.warmup_s + args.cycles * 4 * args.segment_s
    args.log.parent.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(args.url) as ws:
        t0 = time.monotonic()
        with args.log.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "vx", "vy", "yaw", "phase"])
            while True:
                t = time.monotonic() - t0
                if t >= total_s:
                    break
                if t < args.warmup_s:
                    vx = vy = yaw = 0.0
                    phase = "warmup"
                else:
                    idx = int((t - args.warmup_s) // args.segment_s) % 4
                    phase, sx, sy, sz = DIRECTIONS[idx]
                    vx, vy, yaw = args.speed * sx, args.speed * sy, args.speed * sz
                msg = {"vx": vx, "vy": vy, "yaw": yaw, "active": True}
                await ws.send(json.dumps(msg))
                w.writerow([f"{t:.4f}", f"{vx:.4f}", f"{vy:.4f}", f"{yaw:.4f}", phase])
                await asyncio.sleep(dt)
            await ws.send(json.dumps({"vx": 0.0, "vy": 0.0, "yaw": 0.0, "active": False}))
    print(f"[OK] wrote {args.log} ({total_s:.1f}s schedule)")


if __name__ == "__main__":
    asyncio.run(main())
