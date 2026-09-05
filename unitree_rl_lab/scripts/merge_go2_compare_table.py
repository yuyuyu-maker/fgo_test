#!/usr/bin/env python3
"""Merge Go2 play-env step-sweep summaries into one comparison table."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/workspace/fgo_test/unitree_rl_lab")
PIPE = ROOT / "logs/fpo/go2_arch_ablation"
OUT = PIPE / "compare_table.txt"
MD = PIPE / "compare_table.md"

ORDER = [
    "baseline",
    "reflow",
    "reward_aware",
    "kd_only",
    "reflow_teacher_kd",
    "all_ideas_teacher_kd",
]
STEPS = ("64", "32", "16", "8", "4", "1")
NUM = r"[-+]?[0-9]*\.?[0-9]+"
ROW_RE = re.compile(
    rf"^(?P<name>\S+)/(?P<mode>zero|random)\s+"
    rf"(?P<v64>{NUM})\s+(?P<v32>{NUM})\s+(?P<v16>{NUM})\s+"
    rf"(?P<v8>{NUM})\s+(?P<v4>{NUM})\s*(?P<v1>{NUM})\s*$"
)


def parse_file(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        vals = {s: float(m.group(f"v{s}")) for s in STEPS}
        rows[(m.group("name"), m.group("mode"))] = vals
    return rows


def fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:8.1f}"
    return f"{v:8.2f}"


def main() -> None:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for name in (
        "step_sweep_summary_early.txt",
        "step_sweep_summary_teacher_kd.txt",
        "step_sweep_summary.txt",
    ):
        rows.update(parse_file(PIPE / name))
    for log in sorted(PIPE.glob("eval_*.log")):
        rows.update(parse_file(log))

    lines = [
        "# Official Unitree-Go2-Velocity play-env step sweep",
        "# 256 envs, 10 episodes, modes zero/random",
        "",
        f"{'model':<28} " + " ".join(f"{s:>8}" for s in STEPS),
    ]
    md = [
        "# Go2 play-env step comparison",
        "",
        "Protocol: `Unitree-Go2-Velocity` play env, 256 envs, 10 episodes, "
        "Euler steps `{64,32,16,8,4,1}`, `x0` zero vs random.",
        "",
        "| model | 64 | 32 | 16 | 8 | 4 | 1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        for mode in ("zero", "random"):
            key = (name, mode)
            if key not in rows:
                continue
            vals = rows[key]
            label = f"{name}/{mode}"
            lines.append(f"{label:<28} " + " ".join(fmt(vals[s]) for s in STEPS))
            md.append(
                f"| `{label}` | "
                + " | ".join(f"{vals[s]:.2f}" if abs(vals[s]) < 100 else f"{vals[s]:.1f}" for s in STEPS)
                + " |"
            )
        if any((name, m) in rows for m in ("zero", "random")):
            lines.append("")
            md.append("")

    extra = sorted({n for n, _ in rows} - set(ORDER))
    for name in extra:
        for mode in ("zero", "random"):
            key = (name, mode)
            if key not in rows:
                continue
            vals = rows[key]
            label = f"{name}/{mode}"
            lines.append(f"{label:<28} " + " ".join(fmt(vals[s]) for s in STEPS))

    lines.append("Notes:")
    lines.append("- FPO++ baseline and kd_only collapse at 1-step (and kd_only at 4).")
    lines.append("- reflow / reward_aware / full all_ideas_teacher_kd keep 1-step ≈ 64-step.")
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    MD.write_text("\n".join(md) + "\n")
    print(text)
    print(f"wrote {OUT}")
    print(f"wrote {MD}")


if __name__ == "__main__":
    main()
