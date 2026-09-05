#!/usr/bin/env bash
# Download Unitree H1 USD for official Unitree-H1-Velocity.
set -eo pipefail
DEST="${UNITREE_MODEL_DIR:-/dev/shm/unitree_model}"
HF="https://huggingface.co/datasets/unitreerobotics/unitree_model/resolve/main"
REL="H1/h1/usd"
MARKER="${DEST}/${REL}/h1.usd"
BASE="${DEST}/${REL}/configuration/h1_base.usd"

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

echo "[$(date '+%F %T')] downloading H1 USD -> ${DEST}"
fetch "${REL}/h1.usd"
fetch "${REL}/configuration/h1_sensor.usd"
fetch "${REL}/configuration/h1_physics.usd"
fetch "${REL}/configuration/h1_base.usd"
test "$(stat -c%s "$BASE")" -gt 1000000
echo "[ok] ${MARKER}"
ls -lh "$MARKER" "$BASE"
