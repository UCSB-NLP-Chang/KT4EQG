import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.config import load_config
from verifier.data import VerifierProtoDataset, make_collate_fn
from verifier.models import SimpleVerifier
from verifier.losses import info_nce_loss
from verifier.eval import eval_ranking
from verifier.data import load_sibling_map
import random
from typing import Dict, List
import json
import wandb

from torch.utils.data import Subset
import numpy as np

def load_vocab_sizes(output_dir: str):
    concept_vocab_path = os.path.join(output_dir, "concept_vocab.json")

    with open(concept_vocab_path, "r", encoding="utf-8") as f:
        concept2id = json.load(f)

    return len(concept2id), concept2id

import torch
from collections import defaultdict

def compute_ranking_metrics(ranks, ks=(1, 3, 5)):
    """
    ranks: 1-D tensor or list, each element is a 1-based rank.
    """
    if isinstance(ranks, list):
        ranks = torch.tensor(ranks, dtype=torch.long)
    n = ranks.numel()
    metrics = {}
    for k in ks:
        metrics[f"hit@{k}"] = (ranks <= k).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    metrics["mrr"] = mrr
    return metrics


@torch.no_grad()
def eval_on_val(model,
                val_loader,
                device,
                concept_vocab_size: int,
                tau: float = 0.07):
    """
    Run one **global ranking** evaluation on the validation set and return a dict:
      {
        "concept_hit@1": ...,
        "concept_hit@3": ...,
        "concept_mrr": ...,
      }

    Enumerates all concepts and ranks them globally.
    """

    model.eval()

    # ---- Precompute the id vector for all concepts ----
    C = concept_vocab_size
    cand_concepts = torch.arange(C, device=device, dtype=torch.long)  # [C]

    # ---- Re-encode all concepts each epoch, since the encoder changes during training ----
    h_c_all = model.encode_concept(cand_concepts)  # [C, D]

    all_c_ranks = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)          # [B, L]
        attn_mask = batch["attention_mask"].to(device)     # [B, L]
        concept_ids = batch["concept_ids"].to(device)      # [B]
        B = concept_ids.size(0)

        # Encode question text
        h_x = model.encode_text(input_ids, attn_mask)      # [B, D]

        # ---------- Concept-level ranking ----------
        concept_logits = (h_x @ h_c_all.t()) / tau         # [B, C]

        true_c_logits = concept_logits[torch.arange(B), concept_ids]  # [B]
        c_ranks = 1 + (concept_logits > true_c_logits.unsqueeze(1)).sum(dim=1)
        all_c_ranks.append(c_ranks.cpu())

    all_c_ranks = torch.cat(all_c_ranks, dim=0)    # [N_val]

    c_metrics = compute_ranking_metrics(all_c_ranks, ks=(1, 3))

    metrics = {
        "concept_hit@1": c_metrics["hit@1"],
        "concept_hit@3": c_metrics["hit@3"],
        "concept_mrr": c_metrics["mrr"],
    }

    return metrics


def train_verifier():
    cfg = load_config()
    vcfg = cfg.verifier
    train_cfg = vcfg.train
    assert vcfg is not None, "Verifier config not found."

    max_epochs = train_cfg.max_epochs  # e.g. 80
    use_early = getattr(train_cfg, "early_stopping", False)
    patience = getattr(train_cfg, "patience", 5)
    min_delta = getattr(train_cfg, "min_delta", 0.0)

    use_wandb = getattr(train_cfg, "wandb", False)

    if use_wandb:
        wandb.init(
            project=getattr(train_cfg, "wandb_project", "eqg-verifier"),
            config={
                "dataset": vcfg.dataset,
                "batch_size": train_cfg.batch_size,
                "lr": train_cfg.lr,
                "encoder_lr": train_cfg.encoder_lr,
                "use_hard_neg": train_cfg.use_hard_negatives,
            }
        )

    device = torch.device(vcfg.train.device)

    dataset_name = vcfg.dataset
    base_dir = os.path.join(vcfg.output_dir, dataset_name)
    train_path = os.path.join(base_dir, "verifier_proto_train.jsonl")
    val_path = os.path.join(base_dir, "verifier_proto_val.jsonl")

    concept_vocab_size, concept2id = load_vocab_sizes(base_dir)
    tree_path = os.path.join(vcfg.tree_dir, dataset_name, "knowledge_graph.json")
    siblings_by_id = load_sibling_map(tree_path, concept2id)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(vcfg.model.encoder_name)

    model = SimpleVerifier(
        encoder_name=vcfg.model.encoder_name,
        concept_vocab_size=concept_vocab_size,
        proj_dim=vcfg.model.proj_dim,
        freeze_encoder=vcfg.model.freeze_encoder,
    ).to(device)

    if vcfg.model.freeze_encoder:
    # Default: freeze everything
        for p in model.encoder.parameters():
            p.requires_grad = False
    else:
        # Partial unfreeze: only unfreeze the last two encoder layers
        for name, p in model.encoder.named_parameters():
            if (
                "layer.4." in name    # MiniLM-last-2-layers
                or "layer.5." in name
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False

    train_dataset = VerifierProtoDataset(train_path)
    test_dataset = VerifierProtoDataset(val_path)

    # If early stopping is used, split train and val
    if use_early:
        # Split ratio
        train_ratio = 0.9        # 90% train, 10% dev (tunable)
        dev_ratio   = 0.1

        N = len(train_dataset)
        indices = np.arange(N)
        np.random.seed(vcfg.proto.seed)
        np.random.shuffle(indices)

        n_train = int(train_ratio * N)
        train_idx = indices[:n_train]
        dev_idx   = indices[n_train:]

        train_subset = Subset(train_dataset, train_idx)
        dev_subset   = Subset(train_dataset, dev_idx)

        print(f"[Split] Train={len(train_subset)}, Dev={len(dev_subset)}, Total={N}")

    collate_fn = make_collate_fn(tokenizer, vcfg.model.max_length)

    if use_early:

        train_loader = DataLoader(
            train_subset,
            batch_size=vcfg.train.batch_size,
            shuffle=True,
            drop_last=True,    # keeps [B, B] contrastive comparison well-defined
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            dev_subset,
            batch_size=vcfg.train.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=vcfg.train.batch_size,
            shuffle=True,
            drop_last=True,    # keeps [B, B] contrastive comparison well-defined
            collate_fn=collate_fn,
        )
        val_loader = None

    # Separate encoder and head parameters
    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": vcfg.train.lr})
    if encoder_params:
        enc_lr = vcfg.train.encoder_lr or vcfg.train.lr
        param_groups.append({"params": encoder_params, "lr": enc_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=vcfg.train.weight_decay,
    )

    tau = vcfg.train.tau

    best_metric = -1e9
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(base_dir, "simple_verifier_best.pt")

    global_step = 0
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            concept_ids = batch["concept_ids"].to(device)

            h_x, h_c = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                concept_ids=concept_ids,
            )  # [B,D], [B,D]

            if vcfg.train.use_hard_negatives:
                h_c_all, labels = build_hard_negatives(model, h_x, h_c, batch, device, concept_vocab_size, siblings_by_id)   # [B,K,D]

                # logits = h_x dot candidates
                logits = torch.einsum("bd,bkd->bk", h_x, h_c_all) / tau   # [B,K]

                B = concept_ids.size(0)
                K = logits.size(1)  # typically 1 + B + a few hard negatives

                # Index convention for the in-batch region of h_cd_all:
                #   index 0        : explicit positive (c_i, d_i)
                #   index 1..B     : (c_j, d_j) from batch entries j=0..B-1
                #   index > B      : manually constructed hard negatives

                # 1. Find which batch entries share the same concept
                same_c = concept_ids.unsqueeze(1) == concept_ids.unsqueeze(0)  # [B,B]

                # 2. Within the in-batch region [1..B], mask every candidate that shares
                #    the current sample's concept. This includes the diagonal (i,i),
                #    since the explicit positive already sits at index 0.

                neg_mask = torch.zeros_like(logits, dtype=torch.bool)  # [B,K] all False
                neg_mask[:, 1 : 1 + B] = same_c                        # covers index 1..B

                # 3. Set the masked logits to a large negative value (effectively removed from candidates)
                logits = logits.masked_fill(neg_mask, -1e9)


                # in-batch logits: [B, B]
                loss, acc = info_nce_loss(logits, labels, tau)
            else:
                # Baseline: use only in-batch negatives
                logits = (h_x @ h_c.t()) / tau   # [B,B]
                B = concept_ids.size(0)
                same_c = concept_ids.unsqueeze(1) == concept_ids.unsqueeze(0)  # [B,B]

                # Only mask out off-diagonal same-(c,d) negatives
                mask = same_c.clone()
                mask[torch.arange(B), torch.arange(B)] = False  # keep (i,i) positives

                logits = logits.masked_fill(mask, -1e9)

                labels = torch.arange(logits.size(0), device=device)
                loss, acc = info_nce_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            global_step += 1

            if global_step % vcfg.train.log_every == 0:
                avg_loss = running_loss / vcfg.train.log_every
                avg_acc = running_acc / vcfg.train.log_every
                print(
                    f"[Epoch {epoch+1}] step={global_step} "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
                )
                if use_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "acc": acc,
                        "step": global_step,
                        "epoch": epoch
                    })
                running_loss = 0.0
                running_acc = 0.0
        
        if use_early:
            # === At the end of each epoch, run one global ranking eval on val ===
            val_metrics = eval_on_val(
                model,
                val_loader,
                device,
                concept_vocab_size,
                tau=train_cfg.tau,
            )

            concept_hit3 = val_metrics["concept_hit@3"]
            print(f"[Epoch {epoch+1}] val concept_hit@1={val_metrics['concept_hit@1']:.4f}, "
                f"concept_hit@3={concept_hit3:.4f}")
            
            if train_cfg.early_mode == "max":
                if concept_hit3 > best_metric:
                    best_metric = concept_hit3
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                if concept_hit3 < best_metric:
                    best_metric = concept_hit3
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
            if use_wandb:
                wandb.log({
                    "val_concept_hit@3": concept_hit3,
                    "val_concept_hit@1": val_metrics["concept_hit@1"],
                    "val_concept_mrr": val_metrics["concept_mrr"],
                    "epoch": epoch
                })

        if (epoch + 1) % 10 == 0:
            print(f"Evaluating on validation set at epoch {epoch+1}...")
            eval_ranking(model, test_dataset, tokenizer, vcfg, debug_topk=5, debug_samples=5)
            print("Evaluation complete.")

    if use_wandb:
        wandb.finish()

    # Save a checkpoint
    ckpt_path = os.path.join(base_dir, "simple_verifier.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Verifier] Saved model to {ckpt_path}")

def build_hard_negatives(
    model,
    h_x: torch.Tensor,
    h_c_pos: torch.Tensor,
    batch,
    device,
    concept_vocab_size: int,
    siblings_by_id: Dict[int, List[int]],
):
    """
    h_x:      [B, D]
    h_c_pos: [B, D]  # embedding of the positive (c*)
    """
    concept_ids = batch["concept_ids"].to(device)  # [B]

    B = h_x.size(0)

    # -------- in-batch negatives: all (c_j), j=0..B-1 ----------
    # For each sample i, the full row later serves as the candidate set
    h_c_inbatch = h_c_pos.unsqueeze(0).expand(B, B, -1)  # [B, B, D]

    # -------- Hard negative 3: siblings ----------
    sibling_neg_c_list = []
    for i in range(B):
        cid = int(concept_ids[i].item())
        sibs = siblings_by_id.get(cid, None)

        if sibs:
            # Sample one from siblings
            sid = random.choice(sibs)
        else:
            # No siblings -> fall back to a random wrong concept from the full vocab
            # (avoid choosing cid itself; can also reuse the hard_neg_c logic above)
            sid_candidates = [
                j for j in range(concept_vocab_size) if j != cid
            ]
            sid = random.choice(sid_candidates)

        sibling_neg_c_list.append(sid)

    sibling_neg_c = torch.tensor(
        sibling_neg_c_list,
        device=device,
        dtype=torch.long,
    )  # [B]

    h_c_neg_sibling = model.encode_concept(sibling_neg_c)  # [B, D]
    # -------- Concatenate all candidates ----------
    # Order: [pos, in-batch, neg_sibling]
    # pos: [B,1,D]
    # in-batch: [B,B,D]
    # neg_sibling: [B,1,D]
    h_c_all = torch.cat(
        [
            h_c_pos.unsqueeze(1),
            h_c_inbatch,
            h_c_neg_sibling.unsqueeze(1),
        ],
        dim=1,
    )  # [B, K, D], K = 1 + B + 1

    labels = torch.zeros(B, dtype=torch.long, device=device)  # positive index is always 0

    return h_c_all, labels



if __name__ == "__main__":
    train_verifier()
