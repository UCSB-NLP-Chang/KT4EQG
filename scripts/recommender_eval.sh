#!/usr/bin/env bash
set -euo pipefail

# ====== Recommender Eval (no-generation baseline) ======
# Compares practice-round KC selection strategies on the same exam set.
# Question text is a fixed placeholder — isolates the effect of KC selection.

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
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/EQG_Codes}"
cd "${PROJECT_ROOT}"

# ====== Config (override via env vars) ======
# RECOMMENDER_TYPE: random | qwen | ckpt | oracle_vedu | lowest_posterior
RECOMMENDER_TYPE="${RECOMMENDER_TYPE:-random}"
RECOMMENDER_MODEL_NAME="${RECOMMENDER_MODEL_NAME:-Qwen/Qwen3-8B}"
RECOMMENDER_CKPT_PATH="${RECOMMENDER_CKPT_PATH:-}"
RECOMMENDER_DEVICE="${RECOMMENDER_DEVICE:-cuda}"

MODEL_NAME="${MODEL_NAME:-${RECOMMENDER_TYPE}_baseline}"
SPLIT="${SPLIT:-test}"
DATASET="${DATASET:-}"
ROOT_NODE="${ROOT_NODE:-}"
EXAM_SIZE="${EXAM_SIZE:-50}"
EXAM_FIXED_DIFFICULTY="${EXAM_FIXED_DIFFICULTY:-medium}"
PRACTICE_ROUNDS="${PRACTICE_ROUNDS:-20}"
MAX_STUDENTS="${MAX_STUDENTS:--1}"
SEED="${SEED:-42}"
PRACTICE_FIXED_DIFFICULTY="${PRACTICE_FIXED_DIFFICULTY:-medium}"
PRACTICE_RESPONSE_MODE="${PRACTICE_RESPONSE_MODE:-always_correct}"

LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.8}"
LLM_TOP_P="${LLM_TOP_P:-0.9}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-128}"

EVAL_BASE_ROOT="${EVAL_BASE_ROOT:-${PROJECT_ROOT}/output/exam_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_BASE_ROOT}/Eval-Result}"
SHARED_EXAM_ROOT="${SHARED_EXAM_ROOT:-${EVAL_BASE_ROOT}/Eval-Shared/shared_exam_sets}"
SHARED_INITIAL_ROOT="${SHARED_INITIAL_ROOT:-${EVAL_BASE_ROOT}/Eval-Shared/shared_initial_evals}"
EXAM_QUESTION_INFO_PATH="${EXAM_QUESTION_INFO_PATH:-}"

echo "[Recommender-Eval Config]"
echo "  recommender_type=${RECOMMENDER_TYPE}"
echo "  recommender_model_name=${RECOMMENDER_MODEL_NAME}"
echo "  recommender_ckpt_path=${RECOMMENDER_CKPT_PATH:-<none>}"
echo "  split=${SPLIT}"
echo "  dataset=${DATASET:-<from_config>}"
echo "  root_node=${ROOT_NODE:-<from_config>}"
echo "  exam_size=${EXAM_SIZE}"
echo "  practice_rounds=${PRACTICE_ROUNDS}"
echo "  practice_fixed_difficulty=${PRACTICE_FIXED_DIFFICULTY}"
echo "  practice_response_mode=${PRACTICE_RESPONSE_MODE}"
echo "  max_students=${MAX_STUDENTS}"

python eval/recommender_eval.py \
  --model-name "${MODEL_NAME}" \
  --dataset "${DATASET}" \
  --root-node "${ROOT_NODE}" \
  --split "${SPLIT}" \
  --exam-size "${EXAM_SIZE}" \
  --exam-fixed-difficulty "${EXAM_FIXED_DIFFICULTY}" \
  --practice-rounds "${PRACTICE_ROUNDS}" \
  --burn-in-step 10 \
  --max-students "${MAX_STUDENTS}" \
  --seed "${SEED}" \
  --recommender-type "${RECOMMENDER_TYPE}" \
  --recommender-model-name "${RECOMMENDER_MODEL_NAME}" \
  --recommender-ckpt-path "${RECOMMENDER_CKPT_PATH}" \
  --recommender-device "${RECOMMENDER_DEVICE}" \
  --llm-temperature "${LLM_TEMPERATURE}" \
  --llm-top-p "${LLM_TOP_P}" \
  --llm-max-new-tokens "${LLM_MAX_NEW_TOKENS}" \
  --practice-fixed-difficulty "${PRACTICE_FIXED_DIFFICULTY}" \
  --practice-response-mode "${PRACTICE_RESPONSE_MODE}" \
  --output-root "${OUTPUT_ROOT}" \
  --shared-exam-root "${SHARED_EXAM_ROOT}" \
  --shared-initial-root "${SHARED_INITIAL_ROOT}" \
  --exam-question-info-path "${EXAM_QUESTION_INFO_PATH}"
