"""
Stage 1 SFT Data Preparation: Prepare training data with student states.

This script pairs existing questions with RANDOM student states from the appropriate module.
For each question (c, d, x):
1. Determine which module the concept c belongs to
2. Randomly sample a student state from that module
3. Build the prompt using that student state
4. Create training example with forced (c, d) and target question x

This prevents overfitting to specific students while providing realistic context.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

EQG_ROOT = Path(__file__).resolve().parents[1]
if str(EQG_ROOT) not in sys.path:
    sys.path.append(str(EQG_ROOT))

from config.config import load_config
from kt.runtime import KTRuntime
from rl.simple_loop import _leaf_concepts
from verl_bridge.prompt_build import format_student_state_text


# Module configuration for each dataset
DATASET_MODULES = {
    "XES3G5M": ["application_states", "computation_states", "counting_states"],
    "MOOCRadar": ["wine_states", "circuit_states", "education_states"],
}


def get_module_path(dataset_name: str, module_name: str) -> Path:
    """Get the path to a specific module's student states (train subdirectory)."""
    return EQG_ROOT / "data" / "dataset" / dataset_name / module_name / "train"


def load_module_leaf_concepts(
    dataset_name: str, 
    module_name: str,
    cache_dir: Optional[Path] = None
) -> List[str]:
    """
    Load leaf concepts for a module. Uses cache if available.
    
    Args:
        dataset_name: Name of dataset (e.g., "XES3G5M")
        module_name: Name of module (e.g., "application_states")
        cache_dir: Directory to cache results
    
    Returns:
        List of leaf concept names for this module
    """
    if cache_dir is None:
        cache_dir = EQG_ROOT / "data" / "sft_data" / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{module_name}_leaf_concepts.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            leaves = json.load(f)
        print(f"  Loaded {len(leaves)} leaf concepts from cache: {module_name}")
        return leaves

    module_path = get_module_path(dataset_name, module_name)
    if not module_path.exists():
        print(f"  Warning: Module path not found: {module_path}")
        return []
    
    rt = KTRuntime(str(module_path), burn_in_size=10)
    student_ids = list(rt.iter_student_ids())
    
    if not student_ids:
        print(f"  Warning: No students found in {module_name}")
        return []

    sample_graph = rt.load_student_graph(student_ids[0])
    leaves = _leaf_concepts(sample_graph)

    with open(cache_file, 'w') as f:
        json.dump(leaves, f, indent=2)
    
    print(f"  Computed {len(leaves)} leaf concepts for {module_name}")
    return leaves


def build_concept_to_module_mapping(
    dataset_name: str,
    cache_dir: Optional[Path] = None
) -> Dict[str, str]:
    """
    Build a mapping from concept name to module name.
    
    Args:
        dataset_name: Name of dataset
        cache_dir: Directory to cache results
    
    Returns:
        Dict mapping concept_name -> module_name
    """
    if cache_dir is None:
        cache_dir = EQG_ROOT / "data" / "sft_data" / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "concept_to_module.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            mapping = json.load(f)
        print(f"Loaded concept-to-module mapping from cache ({len(mapping)} concepts)")
        return mapping

    print(f"Building concept-to-module mapping for {dataset_name}...")
    modules = DATASET_MODULES.get(dataset_name, [])
    mapping = {}

    for module_name in modules:
        leaves = load_module_leaf_concepts(dataset_name, module_name, cache_dir)
        for concept in leaves:
            if concept in mapping:
                print(f"  Warning: Concept '{concept}' appears in multiple modules")
            mapping[concept] = module_name

    with open(cache_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Built mapping for {len(mapping)} concepts across {len(modules)} modules")
    return mapping


def load_random_student_state(
    dataset_name: str,
    module_name: str,
    leaves: List[str],
    seed: Optional[int] = None
) -> Tuple[str, str]:
    """
    Load a random student's state from the specified module.
    
    Args:
        dataset_name: Name of dataset
        module_name: Name of module
        leaves: Leaf concepts to include in state text
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (student_id, formatted_state_text)
    """
    module_path = get_module_path(dataset_name, module_name)
    rt = KTRuntime(str(module_path), burn_in_size=10)
    
    student_ids = list(rt.iter_student_ids())
    if not student_ids:
        raise ValueError(f"No students found in {module_name}")

    if seed is not None:
        random.seed(seed)
    student_id = random.choice(student_ids)

    student_graph = rt.load_student_graph(student_id)
    state_text = format_student_state_text(student_graph, leaves, precision=4)
    
    return student_id, state_text


def load_question_data(cfg) -> List[Dict]:
    """Load question data from SFT_raw_data/{dataset}/sft_no_figure.jsonl."""
    dataset_name = cfg.verifier.dataset
    question_data_path = EQG_ROOT.parent / "SFT_raw_data" / dataset_name / "sft_no_figure.jsonl"

    if not question_data_path.exists():
        raise FileNotFoundError(f"SFT raw data not found at {question_data_path}")

    # difficulty_name is kept for compatibility with existing processing.
    questions = []
    with open(question_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            q_data = json.loads(line)
            questions.append({
                "qid": q_data.get("qid", "unknown"),
                "text": q_data["text"],
                "concept_name": q_data["concept_name"],
                "difficulty_name": q_data.get("difficulty_name", q_data.get("difficulty", "medium")),
            })

    print(f"Loaded {len(questions)} questions from {question_data_path}")
    return questions


def create_sft_examples_with_random_states(
    questions: List[Dict],
    dataset_name: str,
    concept_to_module: Dict[str, str],
    module_to_leaves: Dict[str, List[str]],
    seed: int = 1234
) -> List[Dict]:
    """
    Create SFT training examples by pairing questions with RANDOM student states.
    
    For each question (c, d, x):
    1. Find which module concept c belongs to
    2. Randomly sample a student state from that module
    3. Build prompt using that state
    4. Create example with forced (c, d) and target x
    
    Args:
        questions: List of question dicts with concept_name, difficulty_name, text
        dataset_name: Name of dataset (e.g., "XES3G5M")
        concept_to_module: Mapping from concept name to module name
        module_to_leaves: Mapping from module name to its leaf concepts
        seed: Random seed
    
    Returns:
        List of training examples
    """
    random.seed(seed)
    examples = []

    print(f"Creating SFT examples with random student states...")

    module_counts = defaultdict(int)
    skipped = 0

    for i, q in enumerate(questions):
        concept = q["concept_name"]

        module = concept_to_module.get(concept)
        if module is None:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: Concept '{concept}' not found in any module, skipping...")
            continue

        module_counts[module] += 1
        leaves = module_to_leaves[module]

        try:
            student_id, state_text = load_random_student_state(
                dataset_name, module, leaves, seed=seed + i
            )
        except Exception as e:
            print(f"  Warning: Could not load student state for {module}: {e}")
            continue

        example = {
            "student_id": student_id,
            "module": module,
            "student_state": state_text,
            "forced_kc": concept,
            "forced_difficulty": q["difficulty_name"],
            "target_question": q["text"],
            "qid": q.get("qid", "unknown"),
        }
        examples.append(example)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(questions)} questions...")
    
    print(f"\n✅ Created {len(examples)} training examples")
    print(f"   Skipped {skipped} questions (concept not found in modules)")
    print(f"   Examples per module:")
    for module, count in sorted(module_counts.items()):
        print(f"     {module}: {count}")
    
    return examples


def save_jsonl(examples: List[Dict], output_path: str):
    """Save examples as JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved {len(examples)} examples to {output_path}")


def main():
    """Main data preparation pipeline."""
    cfg = load_config()
    dataset_name = cfg.verifier.dataset  # e.g., "XES3G5M" or "MOOCRadar"
    
    print("="*80)
    print(f"Stage 1 SFT Data Preparation: {dataset_name}")
    print("="*80)

    print("\n[Step 1/4] Loading question data from SFT_raw_data/{dataset}/sft_no_figure.jsonl...")
    questions = load_question_data(cfg)

    print("\n[Step 2/4] Building concept-to-module mapping...")
    concept_to_module = build_concept_to_module_mapping(dataset_name)

    print("\n[Step 3/4] Loading leaf concepts for each module...")
    modules = DATASET_MODULES.get(dataset_name, [])
    module_to_leaves = {}
    for module in modules:
        leaves = load_module_leaf_concepts(dataset_name, module)
        module_to_leaves[module] = leaves
        print(f"  {module}: {len(leaves)} leaf concepts")

    print("\n[Step 4/4] Creating training examples...")
    all_examples = create_sft_examples_with_random_states(
        questions,
        dataset_name,
        concept_to_module,
        module_to_leaves,
        seed=42
    )

    # Split at prep time (not in dataloader) so the split is reproducible across runs.
    random.seed(42)
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    print(f"\nSplit: {len(train_examples)} train, {len(val_examples)} val")

    output_dir = EQG_ROOT / "data" / "sft_data" / dataset_name
    save_jsonl(train_examples, str(output_dir / "train_with_states.jsonl"))
    save_jsonl(val_examples, str(output_dir / "val_with_states.jsonl"))
    
    print("\n" + "="*80)
    print("✅ Data preparation complete!")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - train_with_states.jsonl")
    print(f"  - val_with_states.jsonl")
    print(f"  - concept_to_module.json (mapping cache)")
    print(f"  - *_leaf_concepts.json (per-module caches)")
    print(f"\nNote: Data is loaded from SFT_raw_data/{{dataset}}/sft_no_figure.jsonl and split 80/20 here.")
    print(f"      Any existing train.jsonl/val.jsonl are NOT used.")


if __name__ == "__main__":
    main()
