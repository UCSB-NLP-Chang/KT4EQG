"""
Stage 1a: Supervised Fine-Tuning with FORCED (c,d) via Gradient Masking

This script implements SFT training with SELECTIVE GRADIENT MASKING:

TRAINING STRATEGY:
1. Full JSON in target: Model generates {"knowledge_concept": "X", "difficulty_level": "Y", "question_text": "..."}
2. Gradient masking: Compute loss on JSON structure + question_text, MASK loss on (c,d) values
3. Result: Model learns JSON generation + question generation, but NOT (c,d) prediction

WHY THIS WORKS:
- Model generates full JSON → gradients flow through structure
- (c,d) values masked → no gradients on forced tokens
- Teaches: "Given (c,d), generate appropriate question with proper JSON"
- Does NOT teach: "Predict what (c,d) should be"

EVALUATION:
- During inference, we use token injection to force (c,d) values
- Model freely generates JSON structure and question_text
- See eval_sft_stage1.py for implementation

This is the "logit manipulation" approach implemented at training time via gradient masking.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

EQG_ROOT = Path(__file__).resolve().parents[1]
if str(EQG_ROOT) not in sys.path:
    sys.path.append(str(EQG_ROOT))

from verl_bridge.chat_template_utils import apply_chat_template_compat
from prompt.prompts import QUESTION_SYSTEM_PROMPT_TRAINABLE, question_prompt_trainable


def _is_medium_only_env() -> bool:
    return bool(int(os.getenv("EQG_MEDIUM_ONLY", os.getenv("MEDIUM_ONLY", "1"))))


@dataclass
class SFTConfig:
    """Configuration for Stage 1 SFT training."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Data
    dataset_name: str = "XES3G5M"  # or "MOOCRadar"
    train_file: str = str(EQG_ROOT / "data/sft_data/XES3G5M/train.jsonl")
    val_file: str = str(EQG_ROOT / "data/sft_data/XES3G5M/val.jsonl")
    max_length: int = 1024
    
    # Training
    output_dir: str = str(EQG_ROOT.parent / "Model" / "stage1_sft_XES3G5M")
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    
    # Forcing strategy
    force_kc_difficulty: bool = True  # Enable forced decoding


class SFTDataset(Dataset):
    """Dataset for Stage 1 SFT with forced (kc, difficulty) pairs."""
    
    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        force_kc_difficulty: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.force_kc_difficulty = force_kc_difficulty
        self.medium_only = _is_medium_only_env()

        self.examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} examples from {data_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Check if this is processed data (with student_state) or raw question data
        if "student_state" in example:
            # Processed format from prepare_sft_data.py
            system_prompt = QUESTION_SYSTEM_PROMPT_TRAINABLE
            user_prompt = question_prompt_trainable(example["student_state"])
            forced_kc = example["forced_kc"]
            forced_difficulty = str(example.get("forced_difficulty", "medium")).strip().lower() or "medium"
            question_text = example["target_question"]
        else:
            # Raw question format (no student state)
            system_prompt = QUESTION_SYSTEM_PROMPT_TRAINABLE
            user_prompt = "\\n\\nGenerate a practice question.\\nRespond with the JSON object described in the instructions."
            forced_kc = example["concept_name"]
            forced_difficulty = str(
                example.get("difficulty_name", example.get("difficulty", "medium"))
            ).strip().lower() or "medium"
            question_text = example["text"]
        
        target_json = {
            "knowledge_concept": forced_kc,
            "question_text": question_text,
        }
        if not self.medium_only:
            target_json["difficulty_level"] = forced_difficulty
        target_str = json.dumps(target_json, ensure_ascii=False)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_str},
        ]

        formatted = apply_chat_template_compat(
            self.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        tokenized = self.tokenizer(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]

        # Compute loss on the assistant response only, with forced fields masked out.
        assistant_start = self._find_assistant_start(formatted, self.tokenizer)

        labels = [-100] * len(input_ids)
        if assistant_start is not None and assistant_start < len(input_ids):
            labels[assistant_start:] = input_ids[assistant_start:]

            # Mask forced kc; also mask forced difficulty when not in medium-only mode.
            kc_tokens = self.tokenizer.encode(forced_kc, add_special_tokens=False)
            for i in range(assistant_start, len(input_ids) - len(kc_tokens) + 1):
                if input_ids[i:i+len(kc_tokens)] == kc_tokens:
                    labels[i:i+len(kc_tokens)] = [-100] * len(kc_tokens)
                    break

            if not self.medium_only:
                diff_tokens = self.tokenizer.encode(forced_difficulty, add_special_tokens=False)
                for i in range(assistant_start, len(input_ids) - len(diff_tokens) + 1):
                    if input_ids[i:i+len(diff_tokens)] == diff_tokens:
                        labels[i:i+len(diff_tokens)] = [-100] * len(diff_tokens)
                        break
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }
    
    def _find_assistant_start(self, formatted_text: str, tokenizer) -> int:
        """Find where assistant response starts in the tokenized sequence.

        Heuristic: assume assistant content begins at roughly the last 30% of the
        sequence. Works reliably enough for Qwen-style chat templates.
        """
        full_ids = tokenizer.encode(formatted_text, add_special_tokens=True)
        return int(len(full_ids) * 0.7)


@dataclass
class DataCollatorForSFT:
    """Custom data collator that pads input_ids, labels, and attention_mask."""
    
    tokenizer: AutoTokenizer
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for f in features:
            input_ids = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels = f["labels"][:max_len]

            padding_length = max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 1 SFT Training")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="XES3G5M",
                       help="Dataset name (XES3G5M or MOOCRadar)")
    parser.add_argument("--train_file", type=str, default=None,
                       help="Path to training data (default: data/sft_data/{dataset}/train_with_states.jsonl)")
    parser.add_argument("--val_file", type=str, default=None,
                       help="Path to validation data (default: data/sft_data/{dataset}/val_with_states.jsonl)")
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-constructed if not provided)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="Learning rate schedule: linear, cosine, constant (default: cosine for better convergence)")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio (increase to 0.15 if using higher lr)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch"],
        help="Save strategy (steps or epoch)",
    )
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch"],
        help="Evaluation strategy (steps or epoch)",
    )
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--run_name", type=str, default="stage1_sft")
    parser.add_argument("--logging_dir", type=str, default=None)

    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--tf32", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--use_fsdp", action="store_true", default=False)
    parser.add_argument("--fsdp", type=str, default="full_shard auto_wrap")
    parser.add_argument("--fsdp_min_num_params", type=int, default=100000000)
    parser.add_argument("--fsdp_layer_cls", type=str, default="")

    parser.add_argument("--force_kc_difficulty", action="store_true", default=True)
    
    return parser.parse_args()


def main():
    """Main training loop for Stage 1 SFT."""
    args = parse_args()

    dataset_name = args.dataset_name
    train_file = args.train_file or str(EQG_ROOT / f"data/sft_data/{dataset_name}/train_with_states.jsonl")
    val_file = args.val_file or str(EQG_ROOT / f"data/sft_data/{dataset_name}/val_with_states.jsonl")
    output_dir = args.output_dir or str(EQG_ROOT.parent / "Model" / f"stage1_sft_{dataset_name}")

    config = SFTConfig(
        model_name=args.model_name,
        dataset_name=dataset_name,
        train_file=train_file,
        val_file=val_file,
        max_length=args.max_length,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        force_kc_difficulty=args.force_kc_difficulty,
    )
    
    print("="*80)
    print("Stage 1a: Supervised Fine-Tuning with Forced Decoding")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Train file: {config.train_file}")
    print(f"Val file: {config.val_file}")
    print(f"Output dir: {config.output_dir}")
    print(f"EQG_MEDIUM_ONLY: {int(_is_medium_only_env())}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} (x{config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps})")
    print("="*80)

    print("\n📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",  # required for training
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    model.resize_token_embeddings(len(tokenizer))

    print(f"✅ Model loaded: {model.config._name_or_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    print("\n📚 Loading datasets...")
    train_dataset = SFTDataset(
        config.train_file,
        tokenizer,
        max_length=config.max_length,
        force_kc_difficulty=config.force_kc_difficulty,
    )
    val_dataset = SFTDataset(
        config.val_file,
        tokenizer,
        max_length=config.max_length,
        force_kc_difficulty=config.force_kc_difficulty,
    )

    data_collator = DataCollatorForSFT(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    
    fsdp_cfg = None
    fsdp_mode = ""
    if args.use_fsdp:
        layer_cls = [x.strip() for x in str(args.fsdp_layer_cls).split(",") if x.strip()]
        fsdp_mode = args.fsdp
        fsdp_cfg = {
            "use_orig_params": False,
            "limit_all_gathers": True,
        }
        # HF TrainingArguments requires exactly one wrap policy style.
        if layer_cls:
            fsdp_cfg["transformer_layer_cls_to_wrap"] = layer_cls
        else:
            fsdp_cfg["min_num_params"] = int(args.fsdp_min_num_params)

    logging_dir = args.logging_dir or str(EQG_ROOT / "tensorboard_log" / f"stage1_sft_{config.dataset_name}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        
        # Logging
        logging_dir=logging_dir,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        
        # Evaluation
        eval_strategy=args.eval_strategy,
        eval_steps=config.eval_steps,
        
        # Saving
        save_strategy=args.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Hardware
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=fsdp_mode,
        fsdp_config=fsdp_cfg,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        
        # Misc
        report_to=["tensorboard"],
        run_name=args.run_name,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\n🚀 Starting training...")
    trainer.train()

    print("\n💾 Saving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {config.output_dir}/final")


if __name__ == "__main__":
    main()
