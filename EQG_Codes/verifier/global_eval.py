import os
import sys
import json
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add project root to sys.path (same as train.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from verifier.data import VerifierProtoDataset, make_collate_fn
from verifier.models import SimpleVerifier


def compute_ranking_metrics(ranks, ks=(1, 3, 5)):
    """
    ranks: List[int], 1-based rank
    """
    n = len(ranks)
    metrics = {}
    for k in ks:
        hit = sum(1 for r in ranks if r <= k) / n
        metrics[f"hit@{k}"] = hit
    mrr = sum(1.0 / r for r in ranks) / n
    metrics["mrr"] = mrr
    return metrics


def run_global_eval(ckpt_path=None, config_path: str | None = None):
    cfg = load_config(config_path or "config/config.yaml")
    vcfg = cfg.verifier
    assert vcfg is not None, "Verifier config not found."

    device = torch.device(vcfg.train.device)

    dataset_name = vcfg.dataset
    base_dir = os.path.join(vcfg.output_dir, dataset_name)
    val_path = os.path.join(base_dir, "verifier_proto_val.jsonl")

    # ----- load vocabs -----
    with open(os.path.join(base_dir, "concept_vocab.json"), "r", encoding="utf-8") as f:
        concept2id = json.load(f)

    id2concept = {v: k for k, v in concept2id.items()}

    all_concepts = sorted(concept2id.values())

    num_concepts = len(all_concepts)

    print("========== Global Ranking Config ==========")
    print(f"Dataset:         {dataset_name}")
    print(f"#Concepts (vocab):     {num_concepts}")
    print("===========================================")

    # ----- build model & tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(vcfg.model.encoder_name)

    model = SimpleVerifier(
        encoder_name=vcfg.model.encoder_name,
        concept_vocab_size=num_concepts,
        proj_dim=vcfg.model.proj_dim,
        freeze_encoder=vcfg.model.freeze_encoder,
    ).to(device)

    if ckpt_path is None:  
        ckpt_path = os.path.join(base_dir, "simple_verifier.pt")
    print(f"[GlobalEval] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ----- dataset & dataloader -----
    val_dataset = VerifierProtoDataset(val_path)
    collate_fn = make_collate_fn(tokenizer, vcfg.model.max_length)

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ----- precompute all concept embeddings -----
    cand_concepts = torch.tensor(all_concepts, device=device)

    with torch.no_grad():
        h_c_all = model.encode_concept(cand_concepts)  # [C, D]

    # ----- accumulators -----
    concept_ranks = []     # 1-based rank for concept

    concept_stats = defaultdict(lambda: {"cnt": 0, "hit1": 0, "hit3": 0})

    concept_freq = Counter()

    n = 0

    for batch in loader:
        n += 1
        q_input_ids = batch["input_ids"].to(device)        # [1, L]
        q_attn = batch["attention_mask"].to(device)
        true_c = batch["concept_ids"].item()

        concept_freq[true_c] += 1

        with torch.no_grad():
            h_x = model.encode_text(q_input_ids, q_attn)   # [1, D]

        # logits over all concepts
        with torch.no_grad():
            logits = (h_x @ h_c_all.t()).squeeze(0)       # [C]

        # ---- concept ranking ----
        sorted_idx = torch.argsort(logits, descending=True)  # [C]

        gt_idx = (cand_concepts == true_c).nonzero(as_tuple=True)[0].item()
        rank_c = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item() + 1  # 1-based
        concept_ranks.append(rank_c)

        # per-concept stats for macro
        concept_stats[true_c]["cnt"] += 1
        if rank_c == 1:
            concept_stats[true_c]["hit1"] += 1
        if rank_c <= 3:
            concept_stats[true_c]["hit3"] += 1

    # ---------------- summary ----------------
    print("\n========== Global Ranking Results ==========")
    print(f"#Validation examples: {n}")

    # micro metrics
    c_metrics = compute_ranking_metrics(concept_ranks, ks=(1, 3))

    print("\n-- Micro (over all samples) --")
    print(f"concept Hit@1: {c_metrics['hit@1']:.4f}, Hit@3: {c_metrics['hit@3']:.4f}, MRR: {c_metrics['mrr']:.4f}")

    # macro over concepts (only concepts that appear in val set)
    seen_concepts = [c for c, st in concept_stats.items() if st["cnt"] > 0]
    macro_hit1_list = []
    macro_hit3_list = []
    for c in seen_concepts:
        st = concept_stats[c]
        macro_hit1_list.append(st["hit1"] / st["cnt"])
        macro_hit3_list.append(st["hit3"] / st["cnt"])

    if seen_concepts:
        macro_hit1 = sum(macro_hit1_list) / len(macro_hit1_list)
        macro_hit3 = sum(macro_hit3_list) / len(macro_hit3_list)
    else:
        macro_hit1 = macro_hit3 = 0.0

    print("\n-- Macro over concepts (seen in val) --")
    print(f"#Seen concepts in val: {len(seen_concepts)} / {num_concepts}")
    print(f"Macro concept Hit@1: {macro_hit1:.4f}")
    print(f"Macro concept Hit@3: {macro_hit3:.4f}")

    # concept frequency stats
    print("\n-- Concept frequency in validation set --")
    if concept_freq:
        total_seen = sum(concept_freq.values())
        print(f"Total concept labels in val: {total_seen}")
        # top-10 frequent concepts
        for cid, cnt in concept_freq.most_common(10):
            name = id2concept.get(cid, f"<id={cid}>")
            print(f"  id={cid:3d} | {name:<30} | count={cnt}")

    print("===========================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (.pt file)"
    )
    args = parser.parse_args()

    run_global_eval(ckpt_path=args.ckpt)
