#!/usr/bin/env bash
# Clone and install official Unitree SDK2 Python into go2_deploy/thirdparty.
# Repo: https://github.com/unitreerobotics/unitree_sdk2_python
#
# Needs Python 3.8–3.12 (3.14 has no cyclonedds wheels). Prefer:
#   /path/to/python3.10 -m venv .venv
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/thirdparty/unitree_sdk2_python"
CDDS="${ROOT}/thirdparty/cyclonedds"
URL="${UNITREE_SDK2_GIT:-https://github.com/unitreerobotics/unitree_sdk2_python.git}"

mkdir -p "${ROOT}/thirdparty"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

PY_VER="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MINOR="$("${PYTHON}" -c 'import sys; print(sys.version_info.minor)')"
if [[ "${PY_MINOR}" -ge 13 ]]; then
  echo "[setup] ERROR: Python ${PY_VER} is too new for cyclonedds==0.10.2."
  echo "  Recreate venv with 3.10, e.g.:"
  echo "    rm -rf .venv"
  echo "    /path/to/python3.10 -m venv .venv"
  echo "    source .venv/bin/activate && pip install -r requirements.txt"
  echo "    bash scripts/setup_unitree_sdk2.sh"
  exit 1
fi

if [[ -d "${DEST}/.git" ]]; then
  echo "[setup] Updating existing clone at ${DEST}"
  git -C "${DEST}" pull --ff-only || true
else
  echo "[setup] Cloning ${URL} -> ${DEST}"
  git clone --depth 1 "${URL}" "${DEST}"
fi

# Build CycloneDDS C lib if pip cannot find a wheel / CMAKE_PREFIX_PATH
if [[ ! -f "${CDDS}/install/lib/libddsc.so" && ! -f "${CDDS}/install/lib/libddsc.so.0" ]]; then
  if ! command -v cmake >/dev/null 2>&1; then
    echo "[setup] ERROR: cmake required to build cyclonedds. Install cmake or add it to PATH."
    exit 1
  fi
  echo "[setup] Cloning/building eclipse-cyclonedds 0.10.x -> ${CDDS}"
  if [[ ! -d "${CDDS}/.git" ]]; then
    git clone --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git "${CDDS}"
  fi
  mkdir -p "${CDDS}/build" "${CDDS}/install"
  cmake -S "${CDDS}" -B "${CDDS}/build" -DCMAKE_INSTALL_PREFIX="${CDDS}/install"
  cmake --build "${CDDS}/build" --target install -j"$(nproc)"
fi

export CYCLONEDDS_HOME="${CDDS}/install"
echo "[setup] CYCLONEDDS_HOME=${CYCLONEDDS_HOME}"
echo "[setup] Installing editable package with ${PYTHON} (${PY_VER})"
"${PYTHON}" -m pip install -e "${DEST}"

echo "[setup] Done. Import check:"
"${PYTHON}" -c "from unitree_sdk2py.go2.sport.sport_client import SportClient; print('SportClient OK', SportClient)"

echo
echo "Next:"
echo "  source ${ROOT}/.venv/bin/activate"
echo "  python -m go2_deploy --mode sport --backend sdk2 --iface <NIC>"
echo "Docs: https://support.unitree.com/home/en/developer/Quick_start"
