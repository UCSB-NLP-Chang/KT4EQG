"""
Stage 1b Dataset: Forced (c,d) prefix for V_align optimization.

Instead of complex logits manipulation, we simply append a forced JSON prefix
to the prompt. The model continues from there, generating the rest of the JSON.

Example prefix:
{
    "knowledge_concept": "Perimeter of rectangle",
    "difficulty_level": "easy",

The model completes:"question_text": "<GENERATED QUESTION>"
}

This ensures:
1. Question is generated conditioned on correct (c,d)
2. Reward V_align computed with same (c,d) as generation
3. No gradient on (c,d) selection (fine for Stage 1b, which focuses on alignment)

Data Source: Loads from SFT JSON files (same as Stage 1a training)
- No KTRuntime needed (no V_edu computation)
- Uses ground truth (c,d) from SFT data as forced values
- One checkpoint per dataset (XES3G5M, MOOCRadar)
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

# Ensure relative config loads work even when launched from outside EQG_Codes.
EQG_ROOT = Path(__file__).resolve().parents[1]
if os.getcwd() != str(EQG_ROOT):
    os.chdir(EQG_ROOT)
if str(EQG_ROOT) not in sys.path:
    sys.path.append(str(EQG_ROOT))

from config.config import load_config
from prompt.prompts import QUESTION_SYSTEM_PROMPT_TRAINABLE, question_prompt_trainable
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl_bridge.chat_template_utils import apply_chat_template_compat


class Stage1bDataset(Dataset):
    """
    Stage 1b RL dataset with forced (c,d) prefix.
    
    Key difference from EQGPromptDataset:
    - Loads from SFT JSON files (same as Stage 1a training)
    - No KTRuntime or student graphs needed (no V_edu computation)
    - Appends forced JSON prefix with ground-truth (c,d) to the prompt
    - Model generates question_text continuation only
    - Passes forced (c,d) through extra_info for reward computation
    """

    def __init__(
        self,
        data_files: str | List[str],
        tokenizer,
        processor,
        config,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list):
            data_files = [data_files]

        self.tokenizer = tokenizer
        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.cfg = load_config()

        eqg_cfg = config.get("eqg", {}) or {}
        self.data_source = eqg_cfg.get("data_source", "eqg_stage1b")
        self.dataset_name = eqg_cfg.get("dataset_name", "XES3G5M")  # XES3G5M or MOOCRadar
        self.seed = eqg_cfg.get("seed", 1234)
        self.medium_only = bool(int(os.getenv("EQG_STAGE1B_MEDIUM_ONLY", "0"))) or bool(eqg_cfg.get("medium_only", False))
        self.verbose_sampling = bool(eqg_cfg.get("verbose_sampling", False))
        self._debug_print_limit = int(eqg_cfg.get("debug_print_n", 5))
        self._debug_printed = 0

        sft_data_path = Path(EQG_ROOT) / "data" / "sft_data" / self.dataset_name / "train_with_states.jsonl"
        if not sft_data_path.exists():
            raise FileNotFoundError(
                f"SFT data not found at {sft_data_path}. "
                f"Available datasets: {list((Path(EQG_ROOT) / 'data' / 'sft_data').glob('*'))}"
            )
        
        self.examples = []
        with open(sft_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    if self.medium_only:
                        diff = str(ex.get("forced_difficulty", "")).strip().lower()
                        if diff != "medium":
                            continue
                    self.examples.append(ex)
        
        mode = "medium-only" if self.medium_only else "all-difficulties"
        print(f"[Stage1bDataset] Loaded {len(self.examples)} examples from {sft_data_path} ({mode})")

        self.dataset_size = len(self.examples)
        if max_samples > 0:
            self.dataset_size = min(self.dataset_size, max_samples)

        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return self.dataset_size

    def _build_forced_prefix(self, kc: str, diff: str) -> str:
        """
        Build the forced JSON prefix that model will continue from.
        
        Format:
        {
            "knowledge_concept": "KC_NAME",
            "difficulty_level": "DIFF",
        
        Model generates: "question_text": "<QUESTION>"
        }
        """
        # In medium-only mode, remove difficulty from model-side output format.
        if self.medium_only:
            prefix = f'''{{"knowledge_concept": "{kc}",'''
        else:
            prefix = f'''{{"knowledge_concept": "{kc}","difficulty_level": "{diff}",'''
        return prefix

    def _encode_prompt_with_prefix(
        self, 
        system_prompt: str, 
        user_prompt: str,
        forced_prefix: str
    ) -> Dict[str, torch.Tensor]:
        """
        Encode prompt with forced JSON prefix appended.
        
        The prefix is added as if it's the model's partial output,
        so model continues generating from there.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_prompt = apply_chat_template_compat(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

        # Append the forced prefix as if the model had already emitted it,
        # so generation continues from there.
        raw_prompt_with_prefix = raw_prompt + forced_prefix
        
        model_inputs = self.tokenizer(
            raw_prompt_with_prefix, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Round-robin over examples with a deterministic shuffle.
        example_idx = idx % len(self.examples)
        if self.seed is not None:
            example_idx = (example_idx + self._rng.randint(0, len(self.examples) - 1)) % len(self.examples)

        example = self.examples[example_idx]

        student_id = example["student_id"]
        student_state = example["student_state"]
        forced_kc = example["forced_kc"]
        forced_diff = str(example.get("forced_difficulty", "medium")).strip().lower() or "medium"
        # target_question is intentionally unused: the model generates its own.

        forced_prefix = self._build_forced_prefix(forced_kc, forced_diff)

        if self.verbose_sampling and self._debug_printed < self._debug_print_limit:
            self._debug_printed += 1
            print(
                f"[Stage1bDataset] student={student_id} "
                f"forced_kc={forced_kc} forced_diff={forced_diff}"
            )

        # Build prompt using student_state (same as SFT training)
        system_prompt = QUESTION_SYSTEM_PROMPT_TRAINABLE
        user_prompt = question_prompt_trainable(student_state)

        encoded = self._encode_prompt_with_prefix(system_prompt, user_prompt, forced_prefix)

        ground_truth = {
            "student_id": student_id,
            "dataset_name": self.dataset_name,
            # Forced (c, d) must be passed through for the reward to be consistent with generation.
            "forced_kc": forced_kc,
            "forced_diff": forced_diff,
        }

        return {
            **encoded,
            "data_source": self.data_source,
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": {
                "prompt_meta": {
                    "system": system_prompt,
                    "user": user_prompt,
                    "forced_prefix": forced_prefix,
                },
                "forced_kc": forced_kc,
                "forced_diff": forced_diff,
            },
        }
