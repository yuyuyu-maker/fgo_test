#!/usr/bin/env bash
# Download official Unitree G1 29-DoF USD (not Isaac Lab's older G1.usd).
# Usage: bash scripts/download_unitree_g1_usd.sh
set -eo pipefail
DEST="${UNITREE_MODEL_DIR:-/dev/shm/unitree_model}"
HF="https://huggingface.co/datasets/unitreerobotics/unitree_model/resolve/main"
REL="G1/29dof/usd/g1_29dof_rev_1_0"
BASE="${DEST}/${REL}/configuration/g1_29dof_rev_1_0_base.usd"
MARKER="${DEST}/${REL}/g1_29dof_rev_1_0.usd"

fetch() {
  local rel="$1"
  local out="${DEST}/${rel}"
  mkdir -p "$(dirname "$out")"
  echo "  get ${rel}"
  curl -L --fail --retry 3 -o "${out}.tmp" "${HF}/${rel}"
  mv "${out}.tmp" "$out"
}

if [[ -f "$BASE" ]] && [[ "$(stat -c%s "$BASE")" -gt 1000000 ]]; then
  echo "[ok] already present: ${MARKER}"
  ls -lh "$MARKER" "$BASE"
  exit 0
fi

echo "[$(date '+%F %T')] downloading G1/29dof USD -> ${DEST}"
fetch "${REL}/g1_29dof_rev_1_0.usd"
fetch "${REL}/configuration/g1_29dof_rev_1_0_sensor.usd"
fetch "${REL}/configuration/g1_29dof_rev_1_0_physics.usd"
fetch "${REL}/configuration/g1_29dof_rev_1_0_base.usd"
test "$(stat -c%s "$BASE")" -gt 1000000
echo "[ok] ${MARKER}"
ls -lh "$MARKER" "$BASE"
