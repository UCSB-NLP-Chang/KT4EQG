import torch
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from verifier.data import VerifierProtoDataset, make_collate_fn
from verifier.models import SimpleVerifier


def eval_ranking(model, val_dataset, tokenizer, vcfg, debug_topk=5, debug_samples=5):
    device = next(model.parameters()).device
    model.eval()

    base_dir = os.path.join(vcfg.output_dir, vcfg.dataset)
    with open(os.path.join(base_dir, "concept_vocab.json"), "r", encoding="utf-8") as f:
        concept2id = json.load(f)

    all_concepts = list(concept2id.values())

    # id -> name mapping
    id2concept = {v: k for k, v in concept2id.items()}

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, vcfg.model.max_length),
    )

    n = 0
    top1_c = top3_c = 0

    for batch in loader:
        n += 1
        q_input_ids = batch["input_ids"].to(device)         # [1, L]
        q_attn = batch["attention_mask"].to(device)
        true_c = batch["concept_ids"].item()

        # encode question once
        with torch.no_grad():
            h_x = model.encode_text(q_input_ids, q_attn)    # [1, D]

        # Build all candidate concepts
        cand_concepts = torch.tensor(all_concepts, device=device)

        with torch.no_grad():
            h_c = model.encode_concept(cand_concepts)  # [C, D]
            logits = (h_x @ h_c.t()).squeeze(0)         # [C]

        # Rank
        sorted_idx = torch.argsort(logits, descending=True)    # [K]

        gt_idx = (cand_concepts == true_c).nonzero(as_tuple=True)[0].item()
        rank_c = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item() + 1

        if rank_c == 1:
            top1_c += 1
        if rank_c <= 3:
            top3_c += 1

        # ---- Debug: print top-k predictions for the first `debug_samples` examples ----
        if n <= debug_samples:
            raw_text = batch["texts"][0]   # requires collate_fn to include "texts"
            print("\n[Example]", n)
            print("Question:", raw_text)
            print("Ground truth:",
                  f"concept={id2concept[true_c]} (id={true_c})")
            print(f"Rank of concept: {rank_c}")

            print(f"Top-{debug_topk} predictions:")
            k = min(debug_topk, sorted_idx.size(0))
            for r in range(k):
                idx = sorted_idx[r].item()
                c_id = cand_concepts[idx].item()
                score = logits[idx].item()
                print(f"  {r+1}. "
                      f"concept={id2concept[c_id]} (id={c_id}), "
                      f"score={score:.4f}")

    print(f"\n[Eval] Num examples: {n}")
    print(f"[Eval] concept top-1: {top1_c / n:.4f}, top-3: {top3_c / n:.4f}")
