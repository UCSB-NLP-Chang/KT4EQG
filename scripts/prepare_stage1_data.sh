#!/usr/bin/env bash
# Prepare Stage 1 SFT training data.
#
# Dataset is read from config/config.yaml (KT.dataset). Edit that file to switch datasets.
#
# Output:
#   data/sft_data/{DATASET}/train_with_states.jsonl
#   data/sft_data/{DATASET}/val_with_states.jsonl
#   data/sft_data/{DATASET}/concept_to_module.json         (cache)
#   data/sft_data/{DATASET}/*_leaf_concepts.json           (cache)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/EQG_Codes"

python verl_bridge/prepare_sft_data.py
