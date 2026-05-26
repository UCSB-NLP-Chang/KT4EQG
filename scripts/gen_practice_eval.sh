#!/usr/bin/env bash
set -euo pipefail

# ====== Oracle-Gen-Verify Practice Eval ======
# Uses oracle KC selection + model generation + verifier re-prediction.

# ====== Environment ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export EQG_MEDIUM_ONLY="${EQG_MEDIUM_ONLY:-1}"
export EQG_PROMPT_SHUFFLE=1
export EQG_PROMPT_HASH_SEED=1

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
# MODEL_PATH accepts either a local checkpoint directory or a HuggingFace model name.
# Local dirs are resolved via --ckpt-path; everything else is passed as --model-path.
MODEL_PATH="${MODEL_PATH:-Gyikoo/KT4EQG-XES3G5M}"

SPLIT="${SPLIT:-test}"
DATASET="${DATASET:-}"
ROOT_NODE="${ROOT_NODE:-}"
EXAM_SIZE="${EXAM_SIZE:-50}"
EXAM_FIXED_DIFFICULTY="${EXAM_FIXED_DIFFICULTY:-medium}"
PRACTICE_ROUNDS="${PRACTICE_ROUNDS:-20}"
MAX_STUDENTS="${MAX_STUDENTS:--1}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
PRACTICE_FIXED_DIFFICULTY="${PRACTICE_FIXED_DIFFICULTY:-medium}"
PRACTICE_RESPONSE_MODE="${PRACTICE_RESPONSE_MODE:-always_correct}"
# Resolve dataset for verifier path: use DATASET env if set, else read from config.yaml
_RESOLVED_DATASET="${DATASET}"
if [[ -z "${_RESOLVED_DATASET}" ]]; then
  _RESOLVED_DATASET="$(grep -A0 '^\s*dataset:' "${PROJECT_ROOT}/config/config.yaml" | head -1 | sed 's/.*dataset:\s*"\([^"]*\)".*/\1/')"
fi
VERIFIER_CKPT_PATH="${VERIFIER_CKPT_PATH:-${REPO_ROOT}/Verifier/${_RESOLVED_DATASET}}"
EXAM_QUESTION_INFO_PATH="${EXAM_QUESTION_INFO_PATH:-}"
FREE_GEN="${FREE_GEN:-0}"
COMPUTE_NLL="${COMPUTE_NLL:-0}"
REF_MODEL_PATH="${REF_MODEL_PATH:-Qwen/Qwen3-8B}"

EVAL_BASE_ROOT="${EVAL_BASE_ROOT:-${PROJECT_ROOT}/output/exam_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_BASE_ROOT}/Eval-Result}"
SHARED_EXAM_ROOT="${SHARED_EXAM_ROOT:-${EVAL_BASE_ROOT}/Eval-Shared/shared_exam_sets}"
SHARED_INITIAL_ROOT="${SHARED_INITIAL_ROOT:-${EVAL_BASE_ROOT}/Eval-Shared/shared_initial_evals}"

if [[ -d "${MODEL_PATH}" ]]; then
  MODEL_ARG="--ckpt-path ${MODEL_PATH}"
  MODEL_SOURCE="local"
else
  MODEL_ARG="--model-path ${MODEL_PATH}"
  MODEL_SOURCE="huggingface"
fi

echo "[Gen-Practice-Eval Config]"
echo "  model_path=${MODEL_PATH} (${MODEL_SOURCE})"
echo "  split=${SPLIT}"
echo "  dataset=${DATASET:-<from_config>}"
echo "  root_node=${ROOT_NODE:-<from_config>}"
echo "  practice_rounds=${PRACTICE_ROUNDS}"
echo "  exam_size=${EXAM_SIZE}"
echo "  exam_fixed_difficulty=${EXAM_FIXED_DIFFICULTY}"
echo "  practice_response_mode=${PRACTICE_RESPONSE_MODE}"
echo "  max_students=${MAX_STUDENTS}"
echo "  max_new_tokens=${MAX_NEW_TOKENS}"
echo "  temperature=${TEMPERATURE}"
echo "  device=${DEVICE}"
echo "  free_gen=${FREE_GEN}"
echo "  compute_nll=${COMPUTE_NLL}"
echo "  ref_model_path=${REF_MODEL_PATH}"

# ====== Run ======
VERIFIER_ARG=""
if [[ -n "${VERIFIER_CKPT_PATH}" ]]; then
  VERIFIER_ARG="--verifier-ckpt-path ${VERIFIER_CKPT_PATH}"
fi

FREE_GEN_ARG=""
if [[ "${FREE_GEN}" == "1" ]]; then
  FREE_GEN_ARG="--free-gen"
fi

python eval/gen_practice_eval.py \
  ${MODEL_ARG} \
  --split "${SPLIT}" \
  --dataset "${DATASET}" \
  --root-node "${ROOT_NODE}" \
  --exam-size "${EXAM_SIZE}" \
  --exam-fixed-difficulty "${EXAM_FIXED_DIFFICULTY}" \
  --practice-rounds "${PRACTICE_ROUNDS}" \
  --burn-in-step 10 \
  --max-students "${MAX_STUDENTS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --practice-fixed-difficulty "${PRACTICE_FIXED_DIFFICULTY}" \
  --practice-response-mode "${PRACTICE_RESPONSE_MODE}" \
  --output-root "${OUTPUT_ROOT}" \
  --shared-exam-root "${SHARED_EXAM_ROOT}" \
  --shared-initial-root "${SHARED_INITIAL_ROOT}" \
  --exam-question-info-path "${EXAM_QUESTION_INFO_PATH}" \
  --ref-model-path "${REF_MODEL_PATH}" \
  --compute-nll "${COMPUTE_NLL}" \
  ${VERIFIER_ARG} \
  ${FREE_GEN_ARG}
