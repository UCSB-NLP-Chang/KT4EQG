#!/usr/bin/env bash
set -euo pipefail

# ====== Re-evaluate exam predictions with KT baselines (BKT / DKT) ======
# Reads an existing eval run's practice trajectories and re-predicts
# exam performance using BKT and/or DKT instead of KT2.

# ====== Environment ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ====== Paths ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../EQG_Codes" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -d "${SCRIPT_DIR}/../../EQG_Codes" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/EQG_Codes}"
cd "${PROJECT_ROOT}"

# ====== Config (override via env vars) ======
RUN_DIR="${RUN_DIR:?Must set RUN_DIR to an existing eval run directory}"
MODULE="${MODULE:-application}"
KT_METHOD="${KT_METHOD:-all}"
DATASET="${DATASET:-XES3G5M}"
BURN_IN_SIZE="${BURN_IN_SIZE:-10}"
DEVICE="${DEVICE:-cuda}"

# DKT-specific
DKT_EPOCHS="${DKT_EPOCHS:-50}"
DKT_HIDDEN_DIM="${DKT_HIDDEN_DIM:-100}"
DKT_LR="${DKT_LR:-0.001}"
DKT_BATCH_SIZE="${DKT_BATCH_SIZE:-64}"
DKT_MAX_SEQ_LEN="${DKT_MAX_SEQ_LEN:-60}"

# BKT-specific
BKT_MAX_SEQ_LEN="${BKT_MAX_SEQ_LEN:-60}"

echo "[Reeval KT Baselines Config]"
echo "  run_dir=${RUN_DIR}"
echo "  module=${MODULE}"
echo "  kt_method=${KT_METHOD}"
echo "  dataset=${DATASET}"
echo "  burn_in_size=${BURN_IN_SIZE}"
echo "  device=${DEVICE}"

# ====== Run ======
python eval/reeval_with_kt_baselines.py \
  --run-dir "${RUN_DIR}" \
  --module "${MODULE}" \
  --kt-method "${KT_METHOD}" \
  --dataset "${DATASET}" \
  --burn-in-size "${BURN_IN_SIZE}" \
  --device "${DEVICE}" \
  --dkt-epochs "${DKT_EPOCHS}" \
  --dkt-hidden-dim "${DKT_HIDDEN_DIM}" \
  --dkt-lr "${DKT_LR}" \
  --dkt-batch-size "${DKT_BATCH_SIZE}" \
  --dkt-max-seq-len "${DKT_MAX_SEQ_LEN}" \
  --bkt-max-seq-len "${BKT_MAX_SEQ_LEN}"
