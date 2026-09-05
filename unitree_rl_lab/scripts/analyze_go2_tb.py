#!/usr/bin/env python3
"""Dump Go2 architecture-ablation TensorBoard scalars (no Isaac)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/workspace/fgo_test/unitree_rl_lab")
OUT = ROOT / "logs/fpo/go2_arch_ablation/quant_tb_summary.txt"

RUNS = {
    "ppo_teacher": ROOT / "logs/rsl_rl/unitree_go2_velocity",
    "fpo_plusplus": ROOT / "logs/fpo/unitree_go2_flat_flow",
    "all_ideas_teacher_kd": ROOT / "logs/fpo/unitree_go2_all_ideas_ppo_teacher",
    "reflow": ROOT / "logs/fpo/unitree_go2_flat_flow_reflow",
    "reflow_teacher_kd": ROOT / "logs/fpo/unitree_go2_reflow_teacher_kd",
    "reward_aware": ROOT / "logs/fpo/unitree_go2_reflow_reward_aware",
    "kd_only": ROOT / "logs/fpo/unitree_go2_kd_only",
}

# Aborted early runs have newer mtimes; pin the completed official-MDP runs.
PREFERRED_RUNS = {
    "ppo_teacher": RUNS["ppo_teacher"] / "2026-09-02_17-20-40_unitree_go2_ppo",
    "fpo_plusplus": RUNS["fpo_plusplus"] / "2026-09-02_20-16-21_fpo_baseline_unitree_mdp",
    "all_ideas_teacher_kd": RUNS["all_ideas_teacher_kd"] / "2026-09-03_01-21-56_all_ideas_ppo_teacher_resume300",
    "reflow": RUNS["reflow"] / "2026-09-03_16-00-14_arch_2026-09-03_15-59-34_reflow",
    "reflow_teacher_kd": RUNS["reflow_teacher_kd"] / "2026-09-03_16-00-58_arch_2026-09-03_15-59-34_reflow_teacher_kd",
    "reward_aware": RUNS["reward_aware"] / "2026-09-03_16-01-45_arch_2026-09-03_15-59-34_reward_aware",
    "kd_only": RUNS["kd_only"] / "2026-09-03_16-02-30_arch_2026-09-03_15-59-34_kd_only",
}

KEYS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/reflow",
    "Loss/teacher_kd",
    "Theory/path_straightness",
]


def latest_run(exp_dir: Path, name: str | None = None) -> Path | None:
    if name and name in PREFERRED_RUNS and PREFERRED_RUNS[name].is_dir():
        return PREFERRED_RUNS[name]
    if not exp_dir.is_dir():
        return None
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not runs:
        return None

    def score(p: Path) -> tuple:
        ckpts = list(p.glob("model_*.pt"))
        iters = []
        for c in ckpts:
            try:
                iters.append(int(c.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        max_iter = max(iters) if iters else -1
        return (max_iter, p.stat().st_mtime)

    return max(runs, key=score)


def last_scalar(acc: EventAccumulator, tag: str):
    if tag not in acc.Tags().get("scalars", []):
        return None
    evs = acc.Scalars(tag)
    if not evs:
        return None
    return evs[-1]


def peak_scalar(acc: EventAccumulator, tag: str):
    if tag not in acc.Tags().get("scalars", []):
        return None
    evs = acc.Scalars(tag)
    if not evs:
        return None
    return max(evs, key=lambda e: e.value)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Go2 architecture ablation — TensorBoard snapshot", ""]
    for name, exp in RUNS.items():
        run = latest_run(exp, name)
        lines.append(f"## {name}")
        if run is None:
            lines.append("missing")
            lines.append("")
            continue
        lines.append(f"run: {run}")
        acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
        acc.Reload()
        for key in KEYS:
            last = last_scalar(acc, key)
            peak = peak_scalar(acc, key) if key == "Train/mean_reward" else None
            if last is None:
                continue
            extra = ""
            if peak is not None:
                extra = f"  peak={peak.value:.3f}@iter{peak.step}"
            lines.append(f"  {key}: last={last.value:.4f}@iter{last.step}{extra}")
        lines.append("")
    sweep = ROOT / "logs/fpo/go2_arch_ablation/step_sweep_summary.txt"
    if sweep.is_file():
        lines.append("## step-sweep (from eval)")
        lines.append(sweep.read_text(errors="replace"))
    OUT.write_text("\n".join(lines))
    print(OUT.read_text())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
