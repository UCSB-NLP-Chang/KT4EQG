import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Dict
from collections import defaultdict

class VerifierProtoDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.examples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "qid": ex["qid"],
            "text": ex["text"],
            "concept_id": ex["concept_id"],
        }

def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        concept_ids = torch.tensor([b["concept_id"] for b in batch], dtype=torch.long)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "concept_ids": concept_ids,
            "texts": texts,
        }

    return collate


def load_sibling_map(subtree_path: str, kc2id: Dict[str, int]) -> Dict[int, List[int]]:
    """
    subtree_path: json path of the subtree
    kc2id: KC -> concept_id
    return: concept_id -> List[sibling_concept_id]
    """
    with open(subtree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    # parent_name -> [child_name, ...]
    parent_to_children = defaultdict(list)
    for node_name, node in tree.items():
        parents = node.get("parents", [])
        for p in parents:
            parent_to_children[p].append(node_name)

    siblings_by_id: Dict[int, List[int]] = {}

    for parent, child_names in parent_to_children.items():
        # Convert names to ids (filter out names not in the vocab)
        child_ids = [kc2id[name] for name in child_names if name in kc2id]
        if len(child_ids) <= 1:
            continue  # With only one child, there are no siblings

        for cid in child_ids:
            sibs = [sid for sid in child_ids if sid != cid]
            if sibs:
                siblings_by_id[cid] = sibs

    return siblings_by_id
