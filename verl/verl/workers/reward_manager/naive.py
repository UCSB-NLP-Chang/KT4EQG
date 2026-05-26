# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_index = int(valid_response_length) - 1
            if data_source == "eqg_stage2":
                prefix_len = _compute_prefix_len_for_mask(
                    response_str, valid_response_ids, self.tokenizer, int(valid_response_length)
                )
                reward_extra_info["response_prefix_len"].append(prefix_len)
                if prefix_len < 1:
                    prefix_len = 1
                reward_index = min(prefix_len, int(valid_response_length)) - 1

            reward_tensor[i, reward_index] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


_MASK_STAT_INTERVAL = int(os.getenv("EQG_MASK_STAT_INTERVAL", "50"))
_MASK_STATS = {
    "total": 0,
    "missing_qtext": 0,
    "fallback_encode": 0,
    "clamped_min": 0,
}


def _maybe_log_mask_stats() -> None:
    if _MASK_STAT_INTERVAL <= 0:
        return
    if _MASK_STATS["total"] % _MASK_STAT_INTERVAL != 0:
        return
    print(
        "[stage2_mask_stats] "
        f"total={_MASK_STATS['total']} "
        f"missing_qtext={_MASK_STATS['missing_qtext']} "
        f"fallback_encode={_MASK_STATS['fallback_encode']} "
        f"clamped_min={_MASK_STATS['clamped_min']}",
        flush=True,
    )


def _find_string_value_end(text: str, key: str) -> int | None:
    key_pos = text.find(key)
    if key_pos == -1:
        return None
    colon_idx = text.find(":", key_pos)
    if colon_idx == -1:
        return None
    idx = colon_idx + 1
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != '"':
        return None
    idx += 1
    while idx < len(text):
        ch = text[idx]
        if ch == '"' and text[idx - 1] != "\\":
            return idx + 1
        idx += 1
    return None


def _map_char_to_token_len(char_index: int, response_ids, tokenizer, fallback_len: int) -> int:
    if char_index <= 0:
        return 1
    try:
        ids = response_ids.tolist() if hasattr(response_ids, "tolist") else list(response_ids)
    except Exception:
        _MASK_STATS["fallback_encode"] += 1
        return fallback_len
    char_count = 0
    for i, tok in enumerate(ids):
        try:
            piece = tokenizer.decode([tok], skip_special_tokens=False)
        except Exception:
            _MASK_STATS["fallback_encode"] += 1
            return fallback_len
        char_count += len(piece)
        if char_count >= char_index:
            return i + 1
    return fallback_len


def _compute_prefix_len_for_mask(
    response_str: str,
    valid_response_ids,
    tokenizer,
    valid_response_length: int,
) -> int:
    keys = ['"question_text"', '"question"']
    key_pos = -1
    for key in keys:
        idx = response_str.find(key)
        if idx != -1 and (key_pos == -1 or idx < key_pos):
            key_pos = idx
    if key_pos == -1:
        _MASK_STATS["total"] += 1
        _MASK_STATS["missing_qtext"] += 1
        _maybe_log_mask_stats()
        return valid_response_length

    colon_idx = response_str.find(":", key_pos)
    if colon_idx == -1:
        _MASK_STATS["total"] += 1
        _MASK_STATS["missing_qtext"] += 1
        _maybe_log_mask_stats()
        return valid_response_length

    prefix_end = colon_idx + 1
    while prefix_end < len(response_str) and response_str[prefix_end].isspace():
        prefix_end += 1
    if prefix_end < len(response_str) and response_str[prefix_end] == '"':
        prefix_end += 1

    prefix_len = _map_char_to_token_len(prefix_end, valid_response_ids, tokenizer, valid_response_length)

    safe_min_end = 0
    for key in ('"knowledge_concept"', '"difficulty_level"', '"difficulty"'):
        end_pos = _find_string_value_end(response_str, key)
        if end_pos:
            safe_min_end = max(safe_min_end, end_pos)
    if safe_min_end > 0:
        safe_min_len = _map_char_to_token_len(safe_min_end, valid_response_ids, tokenizer, valid_response_length)
        if safe_min_len > prefix_len:
            prefix_len = safe_min_len
            _MASK_STATS["clamped_min"] += 1

    if prefix_len < 1:
        prefix_len = 1
    if prefix_len > valid_response_length:
        prefix_len = valid_response_length
    _MASK_STATS["total"] += 1
    _maybe_log_mask_stats()
    return prefix_len
