#!/usr/bin/env python3
"""Build Viser GLB assets for Unitree G1 from official URDF + STL meshes.

Example:
  # after cloning unitree_ros robots/g1_description locally:
  python scripts/build_g1_viser_assets_from_urdf.py \\
    --urdf /path/to/g1_29dof_rev_1_0.urdf \\
    --out viser_assets/unitree_g1_29dof_velocity
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

BODIES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def rpy_to_matrix(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def parse_origin(elem) -> np.ndarray:
    if elem is None:
        return np.eye(4)
    xyz = [float(x) for x in (elem.get("xyz") or "0 0 0").split()]
    rpy = [float(x) for x in (elem.get("rpy") or "0 0 0").split()]
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(rpy)
    t[:3, 3] = xyz
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--mesh-root", type=Path, default=None, help="Defaults to URDF parent dir.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mesh_root = args.mesh_root or args.urdf.parent
    out_mesh = args.out / "meshes"
    out_mesh.mkdir(parents=True, exist_ok=True)

    root = ET.parse(args.urdf).getroot()
    link_visuals: dict[str, list] = {}
    for link in root.findall("link"):
        name = link.get("name")
        items = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            fname = mesh.get("filename") or ""
            if "package://g1_description/" in fname:
                rel = fname.split("package://g1_description/")[-1]
            else:
                rel = fname
            scale = mesh.get("scale")
            sc = np.array([float(x) for x in scale.split()]) if scale else np.ones(3)
            items.append((rel, sc, parse_origin(visual.find("origin"))))
        if items:
            link_visuals[name] = items

    mappings = {}
    info_meshes = {}
    for body in BODIES:
        if body not in link_visuals:
            raise SystemExit(f"body {body} missing in URDF visuals")
        parts = []
        for rel, sc, t_vis in link_visuals[body]:
            path = mesh_root / rel
            if not path.exists():
                path = mesh_root / "meshes" / Path(rel).name
            if not path.exists():
                raise SystemExit(f"mesh file missing for {body}: {rel}")
            m = trimesh.load(str(path), force="mesh")
            if not isinstance(m, trimesh.Trimesh):
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            m.apply_scale(sc)
            m.apply_transform(t_vis)
            parts.append(m)
        combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        combined.export(out_mesh / f"{body}.glb")
        out_name = f"meshes/{body}.glb"
        mappings[f"/World/envs/env_0/Robot/{body}/visuals"] = out_name
        info_meshes[body] = {
            "filename": out_name,
            "vertices": int(len(combined.vertices)),
            "faces": int(len(combined.faces)),
        }
        print(f"OK {body}")

    (args.out / "prim_to_mesh.yaml").write_text(
        yaml.dump(
            {
                "metadata": {
                    "task": "Unitree-G1-29dof-Velocity",
                    "source": str(args.urdf),
                    "description": "G1 29DoF visuals from official URDF",
                },
                "mappings": mappings,
            },
            sort_keys=False,
        )
    )
    (args.out / "scene_hierarchy.yaml").write_text("hierarchy: {}\n")
    (args.out / "extraction_info.yaml").write_text(
        yaml.dump(
            {
                "metadata": {"task": "Unitree-G1-29dof-Velocity", "source": str(args.urdf)},
                "summary": {"total_meshes": len(info_meshes)},
                "meshes": info_meshes,
            },
            sort_keys=False,
        )
    )
    print(f"wrote {args.out} ({len(mappings)} meshes)")


if __name__ == "__main__":
    main()
