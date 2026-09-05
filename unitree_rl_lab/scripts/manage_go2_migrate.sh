#!/usr/bin/env bash
# Manage Go2 FPO migration artifacts under /workspace/fgo_test_migrate (or MIGRATE_DIR).
#
# Usage:
#   bash scripts/manage_go2_migrate.sh status
#   bash scripts/manage_go2_migrate.sh verify
#   bash scripts/manage_go2_migrate.sh pack-one          # one big .tar for local copy
#   bash scripts/manage_go2_migrate.sh unpack-one <tar>  # split back on new machine
#   bash scripts/manage_go2_migrate.sh restore-layout [dest]  # expand code into dest
#   bash scripts/manage_go2_migrate.sh restore-conda [conda_parent]  # unpack env tar
#   bash scripts/manage_go2_migrate.sh clean-staging     # remove unpacked staging dir
set -euo pipefail

MIGRATE_DIR="${MIGRATE_DIR:-/workspace/fgo_test_migrate}"
CODE_TGZ="${MIGRATE_DIR}/go2_fpo_bundle_code.tar.gz"
ENV_TGZ="${MIGRATE_DIR}/isaaclab_fpo_env.tar.gz"
SUMS="${MIGRATE_DIR}/SHA256SUMS"
BUNDLE_DIR="${MIGRATE_DIR}/go2_fpo_bundle"
ONE_TAR="${MIGRATE_DIR}/go2_fpo_migrate_ALL.tar"
ONE_NAME="go2_fpo_migrate_ALL.tar"

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

need() {
  local f
  for f in "$@"; do
    if [[ ! -e "$f" ]]; then
      echo "MISSING: $f" >&2
      exit 1
    fi
  done
}

cmd_status() {
  echo "MIGRATE_DIR=$MIGRATE_DIR"
  echo
  if [[ ! -d "$MIGRATE_DIR" ]]; then
    echo "(directory does not exist)"
    exit 1
  fi
  ls -lh "$MIGRATE_DIR" | sed '1d'
  echo
  if [[ -f "$SUMS" ]]; then
    echo "--- SHA256SUMS ---"
    cat "$SUMS"
  fi
  echo
  local total
  total=$(du -sh "$MIGRATE_DIR" 2>/dev/null | awk '{print $1}')
  echo "total disk: $total"
  if [[ -f "$ONE_TAR" ]]; then
    echo "one-pack: $(ls -lh "$ONE_TAR" | awk '{print $5, $9}')"
  else
    echo "one-pack: (not built — run: $0 pack-one)"
  fi
}

cmd_verify() {
  need "$SUMS"
  echo "Verifying checksums in $MIGRATE_DIR ..."
  (cd "$MIGRATE_DIR" && sha256sum -c SHA256SUMS)
  echo "OK"
}

cmd_pack_one() {
  need "$CODE_TGZ" "$ENV_TGZ" "$SUMS"
  # refresh sums for the two archives
  (cd "$MIGRATE_DIR" && sha256sum go2_fpo_bundle_code.tar.gz isaaclab_fpo_env.tar.gz > SHA256SUMS)
  # Write a short copy hint into the migrate dir
  cat > "$MIGRATE_DIR/README_COPY.txt" << EOF
Go2 FPO migrate ALL pack
========================
Contents of ${ONE_NAME}:
  - go2_fpo_bundle_code.tar.gz   (~123MB) code + ckpts + Viser + MIGRATE.md
  - isaaclab_fpo_env.tar.gz      (~7.5GB) conda isaaclab_fpo
  - SHA256SUMS
  - manage_go2_migrate.sh
  - README_COPY.txt
  - MIGRATE.md (if present)

On your laptop:
  scp user@host:${ONE_TAR} ./

On new server:
  tar -xf ${ONE_NAME}
  bash manage_go2_migrate.sh verify
  bash manage_go2_migrate.sh restore-layout /workspace/fgo_test
  bash manage_go2_migrate.sh restore-conda /tmp/isaaclab_conda
  # details: MIGRATE.md
EOF

  # Ship manager script next to archives
  local self
  self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  cp -a "$self" "$MIGRATE_DIR/manage_go2_migrate.sh"
  chmod +x "$MIGRATE_DIR/manage_go2_migrate.sh"
  [[ -f "$BUNDLE_DIR/MIGRATE.md" ]] && cp -a "$BUNDLE_DIR/MIGRATE.md" "$MIGRATE_DIR/MIGRATE.md"

  echo "Building single archive (store-only, no re-compress): $ONE_TAR"
  local members=(
    go2_fpo_bundle_code.tar.gz
    isaaclab_fpo_env.tar.gz
    SHA256SUMS
    README_COPY.txt
    manage_go2_migrate.sh
  )
  [[ -f "$MIGRATE_DIR/MIGRATE.md" ]] && members+=(MIGRATE.md)

  tar -C "$MIGRATE_DIR" -cf "$ONE_TAR" "${members[@]}"

  ls -lh "$ONE_TAR"
  (cd "$MIGRATE_DIR" && sha256sum "$ONE_NAME" | tee SHA256SUMS.one)
  echo
  echo "Copy to local:"
  echo "  scp $(hostname -f 2>/dev/null || hostname):$ONE_TAR ."
  echo "  # or"
  echo "  rsync -avP $ONE_TAR user@laptop:~/Downloads/"
}

cmd_unpack_one() {
  local src="${1:-}"
  if [[ -z "$src" ]]; then
    echo "Usage: $0 unpack-one /path/to/go2_fpo_migrate_ALL.tar [outdir]" >&2
    exit 1
  fi
  need "$src"
  local out="${2:-$MIGRATE_DIR}"
  mkdir -p "$out"
  echo "Extracting $src -> $out"
  tar -C "$out" -xf "$src"
  ls -lh "$out"
  if [[ -f "$out/SHA256SUMS" ]]; then
    (cd "$out" && sha256sum -c SHA256SUMS)
  fi
}

cmd_restore_layout() {
  local dest="${1:-/workspace/fgo_test}"
  need "$CODE_TGZ"
  echo "Expanding code bundle into $dest (unitree_rl_lab + isaaclab_experiments)"
  mkdir -p "$dest" /tmp/go2_migrate_unpack_$$
  tar -C /tmp/go2_migrate_unpack_$$ -xzf "$CODE_TGZ"
  mkdir -p "$dest"
  # bundle root contains unitree_rl_lab + isaaclab_experiments
  local root
  root=$(find /tmp/go2_migrate_unpack_$$ -maxdepth 2 -type d -name unitree_rl_lab | head -1 | xargs dirname)
  cp -a "$root/unitree_rl_lab" "$dest/"
  cp -a "$root/isaaclab_experiments" "$dest/"
  [[ -f "$root/MIGRATE.md" ]] && cp -a "$root/MIGRATE.md" "$dest/"
  rm -rf /tmp/go2_migrate_unpack_$$
  echo "Done. Layout:"
  ls -la "$dest" | head
}

cmd_restore_conda() {
  local parent="${1:-/tmp/isaaclab_conda}"
  need "$ENV_TGZ"
  echo "Unpacking conda to $parent (must be executable local disk)"
  mkdir -p "$parent"
  tar -C "$parent" -xzf "$ENV_TGZ"
  echo "Expect: $parent/miniconda3_isaaclab_fpo/envs/isaaclab_fpo"
  ls "$parent/miniconda3_isaaclab_fpo/envs" 2>/dev/null || ls "$parent"
  echo "Activate with:"
  echo "  source /workspace/fgo_test/isaaclab_experiments/source_env.sh"
  echo "  # CONDA_ROOT default: /tmp/isaaclab_conda/miniconda3_isaaclab_fpo"
}

cmd_clean_staging() {
  if [[ -d "$BUNDLE_DIR" ]]; then
    echo "Removing staging dir $BUNDLE_DIR (archives kept)"
    rm -rf "$BUNDLE_DIR"
  fi
  echo "Kept:"
  ls -lh "$MIGRATE_DIR" || true
}

main() {
  local cmd="${1:-status}"
  shift || true
  case "$cmd" in
    status|st) cmd_status "$@" ;;
    verify|check) cmd_verify "$@" ;;
    pack-one|pack) cmd_pack_one "$@" ;;
    unpack-one|unpack) cmd_unpack_one "$@" ;;
    restore-layout) cmd_restore_layout "$@" ;;
    restore-conda) cmd_restore_conda "$@" ;;
    clean-staging) cmd_clean_staging "$@" ;;
    -h|--help|help) usage 0 ;;
    *) echo "Unknown command: $cmd" >&2; usage 1 ;;
  esac
}

main "$@"
