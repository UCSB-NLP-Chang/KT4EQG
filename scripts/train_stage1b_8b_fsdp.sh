#!/usr/bin/env bash
# Stage 1b (8B + FSDP): RL alignment training.
#
# Usage: bash scripts/train_stage1b_8b_fsdp.sh [init_model_or_ckpt] [output_dir] [total_epochs] [resume_mode]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
EQG_ROOT="${REPO_ROOT}/EQG_Codes"
CONFIG_PATH="${EQG_ROOT}/config/config.yaml"

IFS=$'\t' read -r CFG_DATASET CFG_ROOT_NODE <<< "$(
  python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
kt = cfg.get("KT", {})
print(f"{kt.get('dataset','')}\t{kt.get('root_node','')}")
PY
)"
if [[ -z "${CFG_DATASET}" || -z "${CFG_ROOT_NODE}" ]]; then
  echo "ERROR: Failed to read KT.dataset/root_node from ${CONFIG_PATH}"
  exit 1
fi
MODULE_PREFIX="$(echo "${CFG_ROOT_NODE%%_*}" | tr '[:upper:]' '[:lower:]')"
LOG_ROOT_BASE="${EQG_ROOT}/tensorboard_log"
LOG_ROOT="${LOG_ROOT_BASE}/${MODULE_PREFIX}_states"

export EQG_MEDIUM_ONLY="${EQG_MEDIUM_ONLY:-1}"
STAGE1B_MEDIUM_ONLY="${STAGE1B_MEDIUM_ONLY:-${EQG_MEDIUM_ONLY}}"

INIT_MODEL_OR_CKPT="${1:-${REPO_ROOT}/Model/stage1_sft_${CFG_DATASET}_8b_fsdp/final}"
OUTPUT_DIR="${2:-${REPO_ROOT}/Model/stage1b_${CFG_DATASET}_8b_fsdp}"
TOTAL_EPOCHS="${3:-3}"
RESUME_MODE="${4:-auto}"
STAGE1B_SEED="${STAGE1B_SEED:-1234}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-4}"
KL_COEF="${KL_COEF:-0.5}"
ENTROPY_COEF="${ENTROPY_COEF:-0.003}"

RL_RESPONSE_LENGTH="${RL_RESPONSE_LENGTH:-256}"
RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-4608}"
RL_ROLLOUT_GPU_MEM_UTIL="${RL_ROLLOUT_GPU_MEM_UTIL:-0.60}"
RL_LOGPROB_MICRO_BATCH="${RL_LOGPROB_MICRO_BATCH:-2}"
RL_REF_LOGPROB_MICRO_BATCH="${RL_REF_LOGPROB_MICRO_BATCH:-1}"
RL_ROLLOUT_MAX_MODEL_LEN="${RL_ROLLOUT_MAX_MODEL_LEN:-4608}"
RL_ROLLOUT_MAX_BATCHED_TOKENS="${RL_ROLLOUT_MAX_BATCHED_TOKENS:-4608}"
RL_ROLLOUT_MAX_NUM_SEQS="${RL_ROLLOUT_MAX_NUM_SEQS:-128}"
RL_ENABLE_CHUNKED_PREFILL="${RL_ENABLE_CHUNKED_PREFILL:-0}"
RL_PPO_MAX_TOKENS_PER_GPU="${RL_PPO_MAX_TOKENS_PER_GPU:-4608}"
RL_ACTOR_OPTIM_OFFLOAD="${RL_ACTOR_OPTIM_OFFLOAD:-1}"
RL_ACTOR_PARAM_OFFLOAD="${RL_ACTOR_PARAM_OFFLOAD:-0}"
RL_REF_PARAM_OFFLOAD="${RL_REF_PARAM_OFFLOAD:-1}"

SFT_DATA_DIR="${EQG_ROOT}/data/sft_data/${CFG_DATASET}"
SFT_TRAIN_FILE="${SFT_DATA_DIR}/train_with_states.jsonl"
if [ ! -f "${SFT_TRAIN_FILE}" ]; then
  echo "ERROR: SFT training data not found at ${SFT_TRAIN_FILE}"
  exit 1
fi
DATASET_SIZE="$(wc -l < "${SFT_TRAIN_FILE}")"
if [ "${DATASET_SIZE}" -le 0 ]; then
  echo "ERROR: Empty SFT training data: ${SFT_TRAIN_FILE}"
  exit 1
fi
STEPS_PER_EPOCH=$(( DATASET_SIZE / TRAIN_BATCH_SIZE ))
if [ "${STEPS_PER_EPOCH}" -le 0 ]; then
  STEPS_PER_EPOCH=1
fi
SAVE_FREQ=$(( STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS ))
TEST_EVERY_STEPS="${TEST_EVERY_STEPS:-0}"

resolve_model_path() {
  local ckpt="$1"
  if [ -d "$ckpt/actor/huggingface" ]; then
    echo "$ckpt/actor/huggingface"
  elif [ -d "$ckpt/final" ]; then
    echo "$ckpt/final"
  elif [ -d "$ckpt/huggingface" ]; then
    echo "$ckpt/huggingface"
  else
    echo "$ckpt"
  fi
}

latest_ckpt_dir() {
  local root="$1"
  ls -d "$root"/global_step_* 2>/dev/null | sort -V | tail -n 1
}

if [[ "${INIT_MODEL_OR_CKPT}" == /* || "${INIT_MODEL_OR_CKPT}" == ./* || "${INIT_MODEL_OR_CKPT}" == ../* ]]; then
  if [ ! -d "${INIT_MODEL_OR_CKPT}" ]; then
    echo "ERROR: init local path not found: ${INIT_MODEL_OR_CKPT}"
    exit 1
  fi
fi

CURRENT_MODEL_PATH="$(resolve_model_path "${INIT_MODEL_OR_CKPT}")"
REF_MODEL_PATH="$(resolve_model_path "${REF_MODEL_OR_CKPT:-${INIT_MODEL_OR_CKPT}}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RL_N_GPUS_PER_NODE="${RL_N_GPUS_PER_NODE:-8}"

if [[ -z "${RL_FSDP_SIZE:-}" ]]; then
  if [ "${RL_N_GPUS_PER_NODE}" -ge 4 ]; then
    RL_FSDP_SIZE="${RL_N_GPUS_PER_NODE}"
  else
    RL_FSDP_SIZE=-1
  fi
fi
RL_USE_ORIG_PARAMS="${RL_USE_ORIG_PARAMS:-1}"
if [[ -z "${ROLLOUT_TP_SIZE:-}" ]]; then
  if [ "${RL_N_GPUS_PER_NODE}" -ge 8 ]; then
    ROLLOUT_TP_SIZE=2
  else
    ROLLOUT_TP_SIZE=1
  fi
fi

to_hydra_bool() {
  local v="$1"
  if [[ "$v" == "1" || "$v" == "true" || "$v" == "True" ]]; then
    echo "true"
  else
    echo "false"
  fi
}
RL_ACTOR_OPTIM_OFFLOAD_BOOL="$(to_hydra_bool "${RL_ACTOR_OPTIM_OFFLOAD}")"
RL_ACTOR_PARAM_OFFLOAD_BOOL="$(to_hydra_bool "${RL_ACTOR_PARAM_OFFLOAD}")"
RL_REF_PARAM_OFFLOAD_BOOL="$(to_hydra_bool "${RL_REF_PARAM_OFFLOAD}")"
RL_ENABLE_CHUNKED_PREFILL_BOOL="$(to_hydra_bool "${RL_ENABLE_CHUNKED_PREFILL}")"

RL_REQUIRED_MAX_SEQ_LEN=$(( RL_MAX_PROMPT_LENGTH + RL_RESPONSE_LENGTH ))
if [ "${RL_PPO_MAX_TOKENS_PER_GPU}" -lt "${RL_REQUIRED_MAX_SEQ_LEN}" ]; then
  RL_PPO_MAX_TOKENS_PER_GPU="${RL_REQUIRED_MAX_SEQ_LEN}"
fi
if [ "${RL_ROLLOUT_MAX_MODEL_LEN}" -lt "${RL_REQUIRED_MAX_SEQ_LEN}" ]; then
  RL_ROLLOUT_MAX_MODEL_LEN="${RL_REQUIRED_MAX_SEQ_LEN}"
fi
if [ "${RL_ROLLOUT_MAX_BATCHED_TOKENS}" -lt "${RL_REQUIRED_MAX_SEQ_LEN}" ]; then
  RL_ROLLOUT_MAX_BATCHED_TOKENS="${RL_REQUIRED_MAX_SEQ_LEN}"
fi

RL_ULYSSES_SP_SIZE="${RL_ULYSSES_SP_SIZE:-1}"
if [ "${RL_ULYSSES_SP_SIZE}" -le 0 ]; then
  RL_ULYSSES_SP_SIZE=1
fi
RL_EFFECTIVE_WORLD=$(( RL_N_GPUS_PER_NODE / RL_ULYSSES_SP_SIZE ))
if [ "${RL_EFFECTIVE_WORLD}" -le 0 ]; then
  RL_EFFECTIVE_WORLD=1
fi
while [ $(( (MINI_BATCH_SIZE * N_SAMPLES) % RL_EFFECTIVE_WORLD )) -ne 0 ]; do
  MINI_BATCH_SIZE=$((MINI_BATCH_SIZE + 1))
done
RL_NORM_MINI=$(( (MINI_BATCH_SIZE * N_SAMPLES) / RL_EFFECTIVE_WORLD ))
if [ "${MICRO_BATCH_SIZE}" -le 0 ] || [ "${MICRO_BATCH_SIZE}" -gt "${RL_NORM_MINI}" ] || [ $(( RL_NORM_MINI % MICRO_BATCH_SIZE )) -ne 0 ]; then
  MICRO_BATCH_SIZE=1
fi

export PYTHONPATH="${VERL_ROOT}:${EQG_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export EQG_REWARD_VERBOSE="${EQG_REWARD_VERBOSE:-1}"
export EQG_REWARD_STAT_INTERVAL="${EQG_REWARD_STAT_INTERVAL:-20}"
export EQG_PROMPT_SHUFFLE="${EQG_PROMPT_SHUFFLE:-1}"
export EQG_PROMPT_HASH_SEED="${EQG_PROMPT_HASH_SEED:-1}"
export RAY_TMPDIR_BASE="${RAY_TMPDIR:-/tmp/ray}"
if [ "${#RAY_TMPDIR_BASE}" -gt 24 ]; then
  echo "[WARN] RAY_TMPDIR is too long (${RAY_TMPDIR_BASE}); fallback to /tmp/ray"
  RAY_TMPDIR_BASE="/tmp/ray"
fi
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
mkdir -p "${RAY_TMPDIR_BASE}" "${OUTPUT_DIR}"

RL_LOG_DIR="${LOG_ROOT}/rl/stage1b_8b"
CYCLE_RAY_TMPDIR="${RAY_TMPDIR_BASE}/s1b$$"
mkdir -p "${CYCLE_RAY_TMPDIR}" "${RL_LOG_DIR}"

echo "==================================="
echo "Stage1b 8B FSDP"
echo "==================================="
echo "Dataset (from config): ${CFG_DATASET}"
echo "Init model/ckpt: ${INIT_MODEL_OR_CKPT}"
echo "Resolved actor model: ${CURRENT_MODEL_PATH}"
echo "Resolved ref model: ${REF_MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Dataset size: ${DATASET_SIZE}"
echo "Total epochs: ${TOTAL_EPOCHS}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Save frequency: ${SAVE_FREQ} steps (every ${SAVE_EVERY_EPOCHS} epochs)"
echo "Test frequency: ${TEST_EVERY_STEPS} steps"
echo "Resume mode: ${RESUME_MODE}"
echo "medium_only (concept-only output): ${EQG_MEDIUM_ONLY}"
echo "Stage1b medium-only: ${STAGE1B_MEDIUM_ONLY}"
echo "RL GPUs: ${RL_N_GPUS_PER_NODE} | CUDA_VISIBLE_DEVICES=${RL_CUDA_VISIBLE_DEVICES}"
echo "RL rollout n=${N_SAMPLES} response_len=${RL_RESPONSE_LENGTH} tp=${ROLLOUT_TP_SIZE}"
echo "==================================="

cd "${VERL_ROOT}"
unset RAY_ADDRESS
unset ip_head
ray stop --force >/dev/null 2>&1 || true

CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES}" \
EQG_MEDIUM_ONLY="${STAGE1B_MEDIUM_ONLY}" \
EQG_STAGE1B_MEDIUM_ONLY="${STAGE1B_MEDIUM_ONLY}" \
RL_LOG_DIR="${RL_LOG_DIR}" \
TENSORBOARD_DIR="${RL_LOG_DIR}" \
RAY_TMPDIR="${CYCLE_RAY_TMPDIR}" \
python -m verl.trainer.main_ppo \
  --config-path "${EQG_ROOT}/verl_bridge" \
  --config-name ppo_stage1b \
  +ray_kwargs.ray_init.address=local \
  +ray_kwargs.ray_init.include_dashboard=false \
  +ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
  +ray_kwargs.ray_init._temp_dir="${CYCLE_RAY_TMPDIR}" \
  trainer.logger='["console","tensorboard"]' \
  trainer.experiment_name="eqg_stage1b_${CFG_DATASET}_8b" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_EVERY_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.n_gpus_per_node="${RL_N_GPUS_PER_NODE}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.use_orig_params="${RL_USE_ORIG_PARAMS}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${RL_FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${RL_ACTOR_PARAM_OFFLOAD_BOOL}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${RL_ACTOR_OPTIM_OFFLOAD_BOOL}" \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.ref.fsdp_config.use_orig_params="${RL_USE_ORIG_PARAMS}" \
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${RL_FSDP_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${RL_REF_PARAM_OFFLOAD_BOOL}" \
  actor_rollout_ref.model.path="${CURRENT_MODEL_PATH}" \
  +actor_rollout_ref.ref.model.path="${REF_MODEL_PATH}" \
  +actor_rollout_ref.ref.model.trust_remote_code=true \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${RL_PPO_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.response_length="${RL_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${RL_ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${RL_ROLLOUT_MAX_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${RL_ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${RL_ENABLE_CHUNKED_PREFILL_BOOL}" \
  actor_rollout_ref.rollout.n="${N_SAMPLES}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${RL_LOGPROB_MICRO_BATCH}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${RL_REF_LOGPROB_MICRO_BATCH}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${RL_ROLLOUT_GPU_MEM_UTIL}" \
  data.eqg.dataset_name="${CFG_DATASET}" \
  +data.eqg.medium_only="${STAGE1B_MEDIUM_ONLY}" \
  data.max_prompt_length="${RL_MAX_PROMPT_LENGTH}" \
  data.eqg.seed="${STAGE1B_SEED}" \
  data.shuffle=true \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.dataloader_num_workers=4 \
  data.eqg.verbose_sampling=false \
  data.custom_cls.path="${EQG_ROOT}/verl_bridge/stage1b_dataset.py" \
  custom_reward_function.path="${EQG_ROOT}/verl_bridge/stage1b_reward.py" \
  "hydra.searchpath=[file://${VERL_ROOT}/verl/trainer/config]" \
  critic.enable=false \
  reward_model.enable=false \
  reward_model.launch_reward_fn_async=true

LATEST_CKPT="$(latest_ckpt_dir "${OUTPUT_DIR}")"
if [ -z "${LATEST_CKPT}" ]; then
  echo "ERROR: No checkpoint found under ${OUTPUT_DIR}"
  exit 1
fi

echo ""
echo "==================================="
echo "Stage1b 8B FSDP complete"
echo "Latest checkpoint: ${LATEST_CKPT}"
echo "==================================="
