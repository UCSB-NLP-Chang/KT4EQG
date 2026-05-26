#!/usr/bin/env bash
# Stage 1a (8B + FSDP): supervised fine-tuning with forced decoding.
#
# Usage: bash scripts/train_stage1a_8b_fsdp.sh [dataset_name] [model_name] [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EQG_ROOT="${REPO_ROOT}/EQG_Codes"

echo "=========================================="
echo "Stage 1a: Supervised Fine-Tuning (8B + FSDP)"
echo "=========================================="

# ====== Environment Variables ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export TOKENIZERS_PARALLELISM=false
export EQG_DISABLE_THINKING="${EQG_DISABLE_THINKING:-1}"

# ====== Training Configuration ======
DATASET_NAME="${1:-MOOCRadar}"
MODEL_NAME="${2:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${3:-${REPO_ROOT}/Model/stage1_sft_${DATASET_NAME}_8b_fsdp}"
LOGGING_DIR="${LOGGING_DIR:-${EQG_ROOT}/tensorboard_log/stage1_sft_${DATASET_NAME}}"
RUN_NAME="${RUN_NAME:-stage1_sft_${DATASET_NAME}_8b_fsdp}"
export EQG_MEDIUM_ONLY="${EQG_MEDIUM_ONLY:-1}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"  # Effective batch size = 32
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# ====== FSDP Configuration (aligned with stage2 8b fsdp settings) ======
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FSDP_MODE="${FSDP_MODE:-full_shard auto_wrap}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-100000000}"
FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-auto}"

detect_fsdp_layer_cls() {
  local model_ref="$1"
  local inspect_path="$model_ref"
  local model_type=""
  local lower_ref

  if [ -d "$inspect_path/actor/huggingface" ]; then
    inspect_path="$inspect_path/actor/huggingface"
  elif [ -d "$inspect_path/final" ]; then
    inspect_path="$inspect_path/final"
  elif [ -d "$inspect_path/huggingface" ]; then
    inspect_path="$inspect_path/huggingface"
  fi

  if [ -f "$inspect_path/config.json" ]; then
    model_type="$(python - "$inspect_path/config.json" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(cfg.get("model_type", ""))
except Exception:
    print("")
PY
)"
  fi

  if [ "$model_type" = "qwen3" ]; then
    echo "Qwen3DecoderLayer"
    return
  fi
  if [ "$model_type" = "qwen2" ]; then
    echo "Qwen2DecoderLayer"
    return
  fi

  lower_ref="$(echo "$model_ref" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower_ref" == *"qwen3"* ]]; then
    echo "Qwen3DecoderLayer"
  elif [[ "$lower_ref" == *"qwen2.5"* || "$lower_ref" == *"qwen2"* ]]; then
    echo "Qwen2DecoderLayer"
  else
    echo "Qwen3DecoderLayer"
  fi
}

DETECTED_FSDP_LAYER_CLS="$(detect_fsdp_layer_cls "$MODEL_NAME")"
RUN_FSDP_LAYER_CLS="$FSDP_LAYER_CLS"
if [ -z "$RUN_FSDP_LAYER_CLS" ] || [ "$RUN_FSDP_LAYER_CLS" = "auto" ]; then
  RUN_FSDP_LAYER_CLS="$DETECTED_FSDP_LAYER_CLS"
fi

echo "Dataset: ${DATASET_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "TensorBoard log dir: ${LOGGING_DIR}"
echo "Run name: ${RUN_NAME}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE} (x${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning rate: ${LEARNING_RATE}"
echo "LR schedule: ${LR_SCHEDULE} (with ${WARMUP_RATIO} warmup)"
echo "Max length: ${MAX_LENGTH}"
echo "Eval strategy: ${EVAL_STRATEGY}"
echo "Save strategy: ${SAVE_STRATEGY}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "FSDP mode: ${FSDP_MODE}"
echo "FSDP min params: ${FSDP_MIN_NUM_PARAMS}"
echo "FSDP layer cls: ${RUN_FSDP_LAYER_CLS} (detected: ${DETECTED_FSDP_LAYER_CLS})"
echo "medium_only (concept-only output): ${EQG_MEDIUM_ONLY}"
echo ""

mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}"

# ====== Change to EQG_Codes directory ======
cd "${EQG_ROOT}"

# ====== Run Training ======
echo "Starting Stage1a 8B FSDP training..."
echo ""

torchrun --nproc_per_node="${NPROC_PER_NODE}" --standalone \
  verl_bridge/train_sft_stage1.py \
  --dataset_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type "${LR_SCHEDULE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_length "${MAX_LENGTH}" \
  --eval_strategy "${EVAL_STRATEGY}" \
  --save_strategy "${SAVE_STRATEGY}" \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_dir "${LOGGING_DIR}" \
  --bf16 \
  --gradient_checkpointing \
  --use_fsdp \
  --fsdp "${FSDP_MODE}" \
  --fsdp_min_num_params "${FSDP_MIN_NUM_PARAMS}" \
  --fsdp_layer_cls "${RUN_FSDP_LAYER_CLS}" \
  --force_kc_difficulty

echo ""
echo "Stage 1a (8B + FSDP) training complete!"
echo "Model saved to: ${OUTPUT_DIR}/final"
