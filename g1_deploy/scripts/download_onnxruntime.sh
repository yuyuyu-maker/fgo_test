#!/usr/bin/env bash
# Download ONNX Runtime 1.22.0 into g1_deploy/thirdparty (matches CMakeLists).
# Tolerates filesystems that cannot create symlinks (copies .so names instead).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${ONNXRUNTIME_VERSION:-1.22.0}"
NAME="onnxruntime-linux-x64-${VER}"
DEST="${ROOT}/thirdparty/${NAME}"
URL="${ONNXRUNTIME_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${NAME}.tgz}"

mkdir -p "${ROOT}/thirdparty"

if [[ -f "${DEST}/lib/libonnxruntime.so.${VER}" ]]; then
  echo "[ort] Already present: ${DEST}"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "[ort] Downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${TMP}/${NAME}.tgz" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TMP}/${NAME}.tgz" "${URL}"
else
  echo "[ort] ERROR: need curl or wget" >&2
  exit 1
fi

echo "[ort] Extracting -> ${DEST}"
# Some workspace FS reject symlinks; extract then materialize .so names as copies.
set +e
tar -xzf "${TMP}/${NAME}.tgz" -C "${TMP}"
set -e

SRC="${TMP}/${NAME}"
if [[ ! -d "${SRC}" ]]; then
  echo "[ort] ERROR: extract failed" >&2
  exit 1
fi

rm -rf "${DEST}"
mkdir -p "${DEST}/include" "${DEST}/lib"
cp -a "${SRC}/include/." "${DEST}/include/"
cp -f "${SRC}/lib/libonnxruntime.so.${VER}" "${DEST}/lib/"
# Materialize sonames (hardlink if possible, else copy)
if ln "${DEST}/lib/libonnxruntime.so.${VER}" "${DEST}/lib/libonnxruntime.so.1" 2>/dev/null; then
  ln "${DEST}/lib/libonnxruntime.so.${VER}" "${DEST}/lib/libonnxruntime.so" 2>/dev/null || \
    cp -f "${DEST}/lib/libonnxruntime.so.${VER}" "${DEST}/lib/libonnxruntime.so"
else
  cp -f "${DEST}/lib/libonnxruntime.so.${VER}" "${DEST}/lib/libonnxruntime.so.1"
  cp -f "${DEST}/lib/libonnxruntime.so.${VER}" "${DEST}/lib/libonnxruntime.so"
fi
for f in LICENSE LICENSE.txt Privacy.md README.md ThirdPartyNotices.txt VERSION_NUMBER; do
  [[ -f "${SRC}/${f}" ]] && cp -f "${SRC}/${f}" "${DEST}/" || true
done

if [[ ! -f "${DEST}/lib/libonnxruntime.so.${VER}" ]]; then
  echo "[ort] ERROR: expected ${DEST}/lib/libonnxruntime.so.${VER}" >&2
  exit 1
fi

echo "[ort] Done: ${DEST}"
echo "[ort] Tip: export LD_LIBRARY_PATH=${DEST}/lib:\${LD_LIBRARY_PATH}"
