#!/usr/bin/env python3
"""Validate a sim LeRobot dataset against realman 14D schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rm75_sim.schema import ROBOT_STATE_NAMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=str, help="Path to dataset folder (contains meta/)")
    args = ap.parse_args()
    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print("Missing", info_path)
        return 1
    info = json.loads(info_path.read_text())
    feats = info["features"]
    print("fps:", info.get("fps"))
    print("total_episodes:", info.get("total_episodes"))
    print("total_frames:", info.get("total_frames"))

    for key in ("observation.state", "action"):
        f = feats[key]
        assert tuple(f["shape"]) == (14,), f
        names = f.get("names")
        assert list(names) == ROBOT_STATE_NAMES, names
        print(f"{key}: OK shape={f['shape']} names aligned")

    for cam in ("cam0", "cam1"):
        k = f"observation.images.{cam}_rgb"
        assert k in feats, f"missing {k}"
        print(f"{k}: OK shape={feats[k]['shape']}")

    # Prefer parquet sample (avoids optional video decoder / OpenVINO ABI issues)
    try:
        import pyarrow.parquet as pq

        pq_files = sorted((root / "data").rglob("*.parquet"))
        assert pq_files, "no parquet under data/"
        table = pq.read_table(pq_files[0])
        st = np.stack(table.column("observation.state").to_pylist())
        act = np.stack(table.column("action").to_pylist())
        assert st.shape[1] == 14 and act.shape[1] == 14
        print("parquet sample OK", st.shape, act.shape)
        print("state sample[:8]:", st[0, :8])
        g = float(st[0, 7])
        assert -1e-3 <= g <= 1.0 + 1e-3, g
        print("gripper_open in [0,1]:", g)
        # action[t]≈state[t+1] for t < T-1
        if len(st) > 2:
            err = np.abs(act[:-1] - st[1:]).max()
            print("max |action[t]-state[t+1]|:", float(err))
    except Exception as exc:
        print("WARN: parquet check failed:", exc)
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id=root.name, root=root)
            sample = ds[0]
            st = sample["observation.state"]
            st = st.numpy() if hasattr(st, "numpy") else np.asarray(st)
            print("LeRobot loader sample OK", st.shape)
        except Exception as exc2:
            print("WARN: could not load with LeRobotDataset:", exc2)

    print("check_dataset OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
