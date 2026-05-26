import os
import sys
import json
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from verifier.models import SimpleVerifier


class VerifierScorer:
    """
    Responsibilities:
    1. Load the verifier model + vocab.
    2. Provide the score_alignment(context, concept, difficulty) interface:
       - context: string (the context the LLM sees)
       - concept: either a text label or an external id
       - difficulty: optional, easy/medium/hard (only used when the checkpoint is difficulty-aware)
    """

    def __init__(self, ckpt_path: str | None = None, temperature: float = 1.0, device: str | None = None):
        cfg = load_config()
        vcfg = cfg.verifier
        assert vcfg is not None, "Verifier config not found."

        self.device = torch.device(device or vcfg.train.device)
        self.temperature = temperature
        self.medium_only = bool(int(os.getenv("EQG_MEDIUM_ONLY", os.getenv("MEDIUM_ONLY", "1"))))
        self.diff2id = {"easy": 0, "medium": 1, "hard": 2}

        dataset_name = vcfg.dataset
        base_dir, resolved_ckpt = self._resolve_base_dir_and_ckpt(vcfg, ckpt_path)

        # ----- vocab -----
        with open(os.path.join(base_dir, "concept_vocab.json"), "r", encoding="utf-8") as f:
            self.concept2id = json.load(f)          # key: concept_text, val: vocab_id

        # Reverse mapping (mainly for debug or external id -> text -> vocab id)
        self.id2concept = {v: k for k, v in self.concept2id.items()}

        # ----- encoder + model -----
        self.tokenizer = AutoTokenizer.from_pretrained(vcfg.model.encoder_name)

        state = torch.load(resolved_ckpt, map_location=self.device)
        self.use_difficulty = ("diff_emb.weight" in state) and ("cd_proj.0.weight" in state)
        diff_vocab_size = int(state["diff_emb.weight"].shape[0]) if "diff_emb.weight" in state else 3
        num_concepts = len(self.concept2id)
        self.model = SimpleVerifier(
            encoder_name=vcfg.model.encoder_name,
            concept_vocab_size=num_concepts,
            proj_dim=vcfg.model.proj_dim,
            freeze_encoder=vcfg.model.freeze_encoder,
            use_difficulty=self.use_difficulty,
            difficulty_vocab_size=diff_vocab_size,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # ===== Precompute concept embeddings (sliced by difficulty if applicable) =====
        all_concept_ids = sorted(self.id2concept.keys())   # 0..C-1
        self.candidates = all_concept_ids
        self.id_to_pos = {cid: i for i, cid in enumerate(self.candidates)}
        cand_concepts = torch.tensor(all_concept_ids, device=self.device)

        if self.use_difficulty:
            self.h_c_all_by_diff: dict[int, torch.Tensor] = {}
            with torch.no_grad():
                for diff_id in self.diff2id.values():
                    diff_tensor = torch.full_like(cand_concepts, diff_id)
                    self.h_c_all_by_diff[diff_id] = self.model.encode_concept(
                        cand_concepts,
                        difficulty_ids=diff_tensor,
                    )  # [C, D]
            self.C = next(iter(self.h_c_all_by_diff.values())).size(0)
        else:
            with torch.no_grad():
                self.h_c_all = self.model.encode_concept(cand_concepts)  # [C, D]
            self.C = self.h_c_all.size(0)

        mode = "kc+diff" if self.use_difficulty else "kc-only"
        print(
            f"[verifier] loaded ckpt={resolved_ckpt} mode={mode} medium_only={int(self.medium_only)}",
            flush=True,
        )

    @staticmethod
    def _resolve_base_dir_and_ckpt(vcfg, ckpt_path: str | None) -> tuple[str, str]:
        """
        Resolve verifier artifact directory and checkpoint path.
        - medium_only=1: prefer <dataset>_KC then <dataset>
        - medium_only=0: prefer <dataset> then <dataset>_KC
        """
        dataset_name = vcfg.dataset
        out_dir = vcfg.output_dir
        rel_ckpt = vcfg.inference.ckpt_path
        medium_only = bool(int(os.getenv("EQG_MEDIUM_ONLY", os.getenv("MEDIUM_ONLY", "1"))))

        if ckpt_path is not None:
            # Explicit ckpt_path keeps old behavior. Infer base_dir from checkpoint location.
            if os.path.isabs(ckpt_path):
                resolved_ckpt = ckpt_path
            else:
                resolved_ckpt = os.path.join(out_dir, dataset_name, ckpt_path)
            return os.path.dirname(resolved_ckpt), resolved_ckpt

        candidates = (
            [f"{dataset_name}_KC", dataset_name, f"{dataset_name}_diff"]
            if medium_only
            else [dataset_name, f"{dataset_name}_diff", f"{dataset_name}_KC"]
        )
        for ds in candidates:
            base = os.path.join(out_dir, ds)
            ckpt = os.path.join(base, rel_ckpt)
            vocab = os.path.join(base, "concept_vocab.json")
            if os.path.isfile(ckpt) and os.path.isfile(vocab):
                return base, ckpt

        # Fallback to original config behavior.
        base = os.path.join(out_dir, dataset_name)
        return base, os.path.join(base, rel_ckpt)

    # ---------- Utilities: normalize a concept to its vocab id ----------

    def _concept_to_vocab_id(self, concept: Any) -> int:
        """
        `concept` can be:
        - a text label (matching a key in concept_vocab.json);
        - a vocab_id (int, present in id2concept);
        - or an "external id" (e.g. the kc_id from question_info). In that case the caller
          must map the external id to a vocab label or vocab_id beforehand.
        """
        if isinstance(concept, str):
            # text label -> vocab id
            return self.concept2id[concept]
        elif isinstance(concept, int):
            # Some kind of id: if it's already a vocab id, use it directly;
            # if it's an external id, the caller must ensure alignment with the vocab id (or map it first).
            if concept in self.id2concept:
                return concept
            else:
                raise KeyError(
                    f"concept id {concept} not in verifier vocab; "
                    f"Please map the external kc id to the verifier vocab id or text label before calling VerifierScorer."
                )
        else:
            raise TypeError(f"Unsupported concept type: {type(concept)}")

    def _difficulty_to_id(self, difficulty: Any) -> int:
        """Map difficulty input to 0/1/2 (easy/medium/hard)."""
        if difficulty is None:
            return self.diff2id["medium"]
        if isinstance(difficulty, int):
            if difficulty in self.diff2id.values():
                return difficulty
            raise KeyError(f"Unsupported difficulty id: {difficulty}")
        diff = str(difficulty).strip().lower()
        if diff in self.diff2id:
            return self.diff2id[diff]
        raise KeyError(f"Unsupported difficulty label: {difficulty}")

    # ---------- encode text ----------

    def _encode_text(self, context: str) -> torch.Tensor:
        enc = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            h_x = self.model.encode_text(input_ids, attn_mask)  # [1, D]
        return h_x

    # ---------- Public interface: (context, concept) -> Valign ----------

    def score_alignment(self, context: str, concept: Any, difficulty: Any = None, **kwargs: Any) -> float:
        """
        Return Valign(x, c) in (0, 1).

        - context: full context the LLM sees
        - concept: text label or vocab id
        - difficulty: optional (easy/medium/hard). For compatibility, `diff=...` is also accepted.
        """
        if "diff" in kwargs and difficulty is None:
            difficulty = kwargs["diff"]
        c_id = self._concept_to_vocab_id(concept)
        d_id = self._difficulty_to_id(difficulty)

        if c_id not in self.id2concept:
            raise KeyError(
                f"(concept_id={c_id}) not in verifier vocab"
            )

        h_x = self._encode_text(context)  # [1, D]

        with torch.no_grad():
            # Logits over all concepts
            h_c_all = self.h_c_all_by_diff[d_id] if self.use_difficulty else self.h_c_all
            logits = (h_x @ h_c_all.t()).squeeze(0)   # [C]

            # -- z-score normalization --
            mean = logits.mean()
            std = logits.std(unbiased=False)
            eps = 1e-6
            z_all = (logits - mean) / (std + eps)           # [K]

            # z for the target concept
            idx = self.id_to_pos[c_id]
            z = z_all[idx] / self.temperature                # scalar

            # sigmoid to get a smoother value in (0, 1)
            prob = torch.sigmoid(z)

        return float(prob.item())


if __name__ == "__main__":
    verifier = VerifierScorer()

    cfg = load_config()
    vcfg = cfg.verifier
    assert vcfg is not None, "Verifier config not found."

    dataset_name = vcfg.dataset
    base_dir = os.path.join(vcfg.output_dir, dataset_name)
    val_path = os.path.join(base_dir, "verifier_proto_val.jsonl")
    all_data = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
    
    # random sample 100 data
    import random
    positive_samples = random.sample(all_data, 100)

    # create 100 negative samples (concept is incorrect)
    negative_samples = []
    for data in positive_samples:
        concept_candidates = [c for c in verifier.concept2id.keys() if c != data["concept_name"]]
        concept = random.choice(concept_candidates)
        negative_samples.append({"text": data["text"], "concept_name": concept})
 
    # evalute the average score and variance of each sample in positive_samples, negative_samples
    import numpy as np
    positive_scores = []
    negative_scores = []
    for sample in positive_samples:
        positive_scores.append(verifier.score_alignment(sample["text"], sample["concept_name"]))
    for sample in negative_samples:
        negative_scores.append(verifier.score_alignment(sample["text"], sample["concept_name"]))
    print(f"Average score of positive samples: {np.mean(positive_scores)}")
    print(f"Variance of positive samples: {np.var(positive_scores)}")
    print(f"Average score of negative samples: {np.mean(negative_scores)}")
    print(f"Variance of negative samples: {np.var(negative_scores)}")
