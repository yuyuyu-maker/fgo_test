# thirdparty (local clones — not committed)

These directories are installed on the machine and **ignored by git** (each upstream has its own `.git`):

| Path | Source | Install |
|------|--------|---------|
| `cyclonedds/` | Eclipse Cyclone DDS | `bash scripts/setup_unitree_sdk2.sh` |
| `unitree_sdk2_python/` | Unitree SDK2 Python | same script |
| `unitree_rl_lab/` (optional) | Unitree RL Lab (C++ deploy) | `bash scripts/clone_unitree_rl_lab.sh` |

Do not `git add` these trees; clones of this repo will not include them.
