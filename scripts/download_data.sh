#!/usr/bin/env bash
# Download and extract the KT4EQG data bundle from HuggingFace into EQG_Codes/data/.
#
# Usage:
#   bash scripts/download_data.sh              # default repo: Gyikoo/KT4EQG-data
#   HF_REPO=user/custom bash scripts/download_data.sh
#
# Requires: huggingface_hub (`pip install huggingface_hub`) and standard tar.

set -euo pipefail

HF_REPO="${HF_REPO:-Gyikoo/KT4EQG-data}"
TARBALL="${TARBALL:-data.tar.gz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/EQG_Codes"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.data_cache}"

echo "[download_data] HF repo:   ${HF_REPO}"
echo "[download_data] Tarball:   ${TARBALL}"
echo "[download_data] Extract to: ${TARGET_DIR}/data/"
mkdir -p "${CACHE_DIR}" "${TARGET_DIR}"

if [[ -d "${TARGET_DIR}/data/dataset" && -n "$(ls -A "${TARGET_DIR}/data/dataset" 2>/dev/null)" ]]; then
  echo "[download_data] ${TARGET_DIR}/data/dataset already exists and is non-empty."
  read -r -p "Overwrite? [y/N] " ans
  if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
    echo "[download_data] Aborted."
    exit 0
  fi
fi

echo "[download_data] Fetching ${TARBALL} from ${HF_REPO} ..."
hf download "${HF_REPO}" "${TARBALL}" \
  --repo-type dataset \
  --local-dir "${CACHE_DIR}"

echo "[download_data] Extracting into ${TARGET_DIR}/ ..."
tar -xzf "${CACHE_DIR}/${TARBALL}" -C "${TARGET_DIR}"

echo "[download_data] Done. Data is at ${TARGET_DIR}/data/"
echo "[download_data] (You can delete ${CACHE_DIR} to reclaim disk space.)"
