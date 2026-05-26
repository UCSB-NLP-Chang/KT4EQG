import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple

class SimpleVerifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        concept_vocab_size: int,
        proj_dim: int,
        freeze_encoder: bool = True,
        use_difficulty: bool = False,
        difficulty_vocab_size: int = 3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.use_difficulty = bool(use_difficulty)

        hidden_size = self.encoder.config.hidden_size

        # Project text features to proj_dim
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.Tanh(),
        )

        # concept embedding
        self.concept_emb = nn.Embedding(concept_vocab_size, proj_dim)
        if self.use_difficulty:
            # difficulty embedding + projection for (concept, difficulty) composition.
            self.diff_emb = nn.Embedding(int(difficulty_vocab_size), proj_dim)
            self.cd_proj = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.Tanh(),
            )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, D]"""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token (mean pooling is another option)
        cls = out.last_hidden_state[:, 0, :]   # [B, hidden_size]
        h_x = self.text_proj(cls)              # [B, proj_dim]
        return h_x

    def encode_concept(
        self,
        concept_ids: torch.Tensor,
        difficulty_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B, D]"""
        e_c = self.concept_emb(concept_ids)    # [B, D]
        if not self.use_difficulty:
            return e_c

        if difficulty_ids is None:
            # Default to medium (easy=0, medium=1, hard=2) for backward compatibility.
            difficulty_ids = torch.full_like(concept_ids, 1)
        elif difficulty_ids.ndim == 0:
            difficulty_ids = difficulty_ids.expand_as(concept_ids)

        e_d = self.diff_emb(difficulty_ids)    # [B, D]
        return self.cd_proj(torch.cat([e_c, e_d], dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        concept_ids: torch.Tensor,
        difficulty_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            h_x:  [B, D]
            h_c: [B, D]
        """
        h_x = self.encode_text(input_ids, attention_mask)
        h_c = self.encode_concept(concept_ids, difficulty_ids=difficulty_ids)
        return h_x, h_c
