#!/usr/bin/env bash
# Download the pretrained verifier checkpoints (one per dataset) from HuggingFace
# into Verifier/.
#
# Usage:
#   bash scripts/download_verifier.sh              # default repo: Gyikoo/KT4EQG-verifier
#   HF_REPO=user/custom bash scripts/download_verifier.sh
#
# Requires: huggingface_hub (`pip install huggingface_hub`).

set -euo pipefail

HF_REPO="${HF_REPO:-Gyikoo/KT4EQG-verifier}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/Verifier"

echo "[download_verifier] HF repo:    ${HF_REPO}"
echo "[download_verifier] Target dir: ${TARGET_DIR}"

if [[ -d "${TARGET_DIR}" && -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]]; then
  echo "[download_verifier] ${TARGET_DIR} already exists and is non-empty."
  read -r -p "Overwrite? [y/N] " ans
  if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
    echo "[download_verifier] Aborted."
    exit 0
  fi
fi

mkdir -p "${TARGET_DIR}"
hf download "${HF_REPO}" \
  --repo-type model \
  --local-dir "${TARGET_DIR}"

echo "[download_verifier] Done. Checkpoints at ${TARGET_DIR}/"
