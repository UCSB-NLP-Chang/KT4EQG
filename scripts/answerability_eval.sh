#!/usr/bin/env bash
# Post-hoc answerability check over a gen_outputs.csv produced by
# gen_practice_eval.sh. Uses vLLM to batch-classify whether each generated
# question is well-posed.
#
# Requires vllm (not in requirements.txt): `pip install vllm`.
# vLLM by default grabs ~90% of GPU memory — run this AFTER gen_practice_eval
# has fully released the GPU, and on a GPU not shared with other processes.

set -euo pipefail

# ====== Environment ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

# ====== Paths ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/EQG_Codes}"

# ====== Config (override via env vars) ======
MODEL="${MODEL:-Qwen/Qwen3-4B}"
INPUT_CSV="${INPUT_CSV:-}"
OUTDIR="${OUTDIR:-}"
QUESTION_COL="${QUESTION_COL:-}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-256}"

if [[ -z "${INPUT_CSV}" ]]; then
  cat >&2 <<EOF
[answerability_eval] INPUT_CSV is required.

Example:
  INPUT_CSV=EQG_Codes/output/exam_eval/Eval-Result/<run>/<module>/gen_outputs.csv \\
    bash scripts/answerability_eval.sh

Optional overrides: MODEL, OUTDIR, QUESTION_COL, TP, MAX_MODEL_LEN, MAX_TOKENS.
EOF
  exit 1
fi

# Resolve relative paths before cd'ing into PROJECT_ROOT.
if [[ "${INPUT_CSV}" != /* ]]; then
  INPUT_CSV="$(realpath -m "${INPUT_CSV}")"
fi
if [[ -n "${OUTDIR}" && "${OUTDIR}" != /* ]]; then
  OUTDIR="$(realpath -m "${OUTDIR}")"
fi
# Default OUTDIR: sibling 'answerability' folder next to the input CSV
if [[ -z "${OUTDIR}" ]]; then
  OUTDIR="$(dirname "${INPUT_CSV}")/answerability"
fi

cd "${PROJECT_ROOT}"

echo "[Answerability-Eval Config]"
echo "  model=${MODEL}"
echo "  input_csv=${INPUT_CSV}"
echo "  outdir=${OUTDIR}"
echo "  question_col=${QUESTION_COL:-<auto>}"
echo "  tp=${TP}"
echo "  max_model_len=${MAX_MODEL_LEN}"
echo "  max_tokens=${MAX_TOKENS}"

QUESTION_COL_ARG=""
if [[ -n "${QUESTION_COL}" ]]; then
  QUESTION_COL_ARG="--question_col ${QUESTION_COL}"
fi

python eval/answerability-edu.py \
  --model "${MODEL}" \
  --input_csv "${INPUT_CSV}" \
  --outdir "${OUTDIR}" \
  --tp "${TP}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --max_tokens "${MAX_TOKENS}" \
  ${QUESTION_COL_ARG}
