# KT4EQG: Knowledge Tracing for Educational Question Generation

This repository contains the official implementation of the paper:

[KT4EQG: Personalized Exercise Question Generation via Knowledge Tracing](http://arxiv.org/abs/2605.23933)

KT4EQG is a personalized educational question generation framework that produces effective practice questions for individual students under the guidance of a knowledge tracing (KT) model. The KT model tracks each student's knowledge state from their historical performance and selects the concept whose practice is expected to maximize overall mastery improvement; an LLM-based generator is then trained to produce a question faithfully grounded in the selected concept, with a contrastive verifier enforcing question–concept alignment. 

## Code Architecture

```
KT4EQG/
├── EQG_Codes/              # Main source code
│   ├── config/             # Configuration
│   ├── data/               # Datasets
│   ├── kt/                 # Knowledge tracing model
│   ├── verifier/           # Question-KC alignment verifier
│   ├── value/              # Reward functions
│   ├── verl_bridge/        # SFT & GRPO training entry points, datasets, reward wrappers
│   ├── eval/               # Evaluation
│   ├── prompt/             # Prompt templates for question generation
│   └── models/             # Base LLM wrapper
├── scripts/                # Shell scripts for training and evaluation
├── Verifier/               # Verifier checkpoints
└── verl/                   # VERL RL training framework
```

## Setup

```bash
conda create -n kt4eqg python=3.10 -y
conda activate kt4eqg
pip install -r requirements.txt
```

## Data & Verifier Setup

```bash
bash scripts/download_data.sh
bash scripts/download_verifier.sh
```

## Datasets and Modules

| Dataset     | Modules (`root_node` values)                                           |
|-------------|------------------------------------------------------------------------|
| `XES3G5M`   | `Application_Module`, `Computation_Module`, `Counting_Module`          |
| `MOOCRadar` | `Wine_Knowledge`, `Circuit_Design`, `Education_Theory`                 |

To switch dataset/module, edit `KT.dataset` and `KT.root_node` in [EQG_Codes/config/config.yaml](EQG_Codes/config/config.yaml) before running any script.

## Released Checkpoints

KT4EQG generator checkpoints are hosted on HuggingFace (one per dataset):

| Dataset     | HuggingFace ID                                                                            |
|-------------|-------------------------------------------------------------------------------------------|
| `XES3G5M`   | [`Gyikoo/KT4EQG-XES3G5M`](https://huggingface.co/Gyikoo/KT4EQG-XES3G5M)                   |
| `MOOCRadar` | [`Gyikoo/KT4EQG-MOOCRadar`](https://huggingface.co/Gyikoo/KT4EQG-MOOCRadar)               |

## Training

### Stage 1a: Supervised Fine-Tuning (SFT)

```bash
bash scripts/train_stage1a_8b_fsdp.sh <dataset_name> <base_model> <output_dir>
```

Example:

```bash
bash scripts/train_stage1a_8b_fsdp.sh XES3G5M Qwen/Qwen3-8B ./Model/stage1a_XES3G5M_8b_fsdp
```

### Stage 1b: RL Alignment Training (PPO)

```bash
bash scripts/train_stage1b_8b_fsdp.sh <sft_checkpoint> <output_dir> <total_epochs> <resume_mode>
```

Example:

```bash
bash scripts/train_stage1b_8b_fsdp.sh \
  ./Model/stage1a_XES3G5M_8b_fsdp/final \
  ./Model/stage1b_XES3G5M_8b_fsdp \
  5 auto
```

## Evaluation

Before every run, edit `KT.dataset` and `KT.root_node` in [EQG_Codes/config/config.yaml](EQG_Codes/config/config.yaml) to the pair to evaluate.

### Main Evaluation

Default `MODEL_PATH` is `Gyikoo/KT4EQG-XES3G5M`. The script auto-detects local vs. HuggingFace.

```bash
# Default: loads Gyikoo/KT4EQG-XES3G5M from HuggingFace
bash scripts/gen_practice_eval.sh

# Override with your own local Stage-1b checkpoint
MODEL_PATH=./Model/stage1b_XES3G5M_8b_fsdp/final \
  bash scripts/gen_practice_eval.sh

# Or a different HuggingFace model
MODEL_PATH=Gyikoo/KT4EQG-MOOCRadar \
  bash scripts/gen_practice_eval.sh
```

### Alternative KT Models (BKT / DKT)

Take an existing Main Evaluation run and re-predict exam performance using BKT or DKT in place of the default KT2 model. 

```bash
RUN_DIR=./EQG_Codes/output/exam_eval/Eval-Result/<your-run> \
MODULE=Application_Module \
DATASET=XES3G5M \
  bash scripts/reeval_kt_baselines.sh
```

### Answerability Check

```bash
INPUT_CSV=./EQG_Codes/output/exam_eval/Eval-Result/<your-run>/<module>/gen_outputs.csv \
  bash scripts/answerability_eval.sh
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.