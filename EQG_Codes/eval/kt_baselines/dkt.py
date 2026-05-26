"""Deep Knowledge Tracing (DKT) — KC-level GRU model.

Input at each time step: one-hot encoding of (kc_id, correct).
    - If correct: position kc_id is 1          (first num_kcs dims)
    - If incorrect: position num_kcs + kc_id is 1  (next num_kcs dims)
Output: sigmoid probability for each KC (P(correct) per KC).

Reference: Piech et al., "Deep Knowledge Tracing", NeurIPS 2015.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class DKTModel(nn.Module):
    def __init__(self, num_kcs: int, hidden_dim: int = 100, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_kcs = num_kcs
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * num_kcs
        self.gru = nn.GRU(
            self.input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_kcs)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, 2*num_kcs) input tensor.
            lengths: (batch,) actual lengths for packing.
            hidden: optional initial hidden state.

        Returns:
            pred: (batch, seq_len, num_kcs) predicted P(correct) per KC.
            hidden: final hidden state.
        """
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, hidden = self.gru(packed, hidden)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, hidden = self.gru(x, hidden)
        pred = torch.sigmoid(self.fc(out))
        return pred, hidden


# ------------------------------------------------------------------
# Data encoding
# ------------------------------------------------------------------

def encode_sequence(
    seq: List[Tuple[int, int]],
    num_kcs: int,
) -> torch.Tensor:
    """Encode [(kc_idx, correct), ...] to (seq_len, 2*num_kcs) tensor."""
    T = len(seq)
    x = torch.zeros(T, 2 * num_kcs)
    for t, (kc_idx, correct) in enumerate(seq):
        if correct:
            x[t, kc_idx] = 1.0
        else:
            x[t, num_kcs + kc_idx] = 1.0
    return x


def encode_target(
    seq: List[Tuple[int, int]],
    num_kcs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create target tensor and mask for next-step prediction.

    For input at time t, target is (kc_{t+1}, correct_{t+1}).
    Returns:
        target: (seq_len, num_kcs) — correct label at the target KC position.
        mask: (seq_len, num_kcs) — 1.0 at the target KC position, 0.0 elsewhere.
    """
    T = len(seq)
    target = torch.zeros(T, num_kcs)
    mask = torch.zeros(T, num_kcs)
    for t in range(T - 1):
        kc_next, correct_next = seq[t + 1]
        target[t, kc_next] = float(correct_next)
        mask[t, kc_next] = 1.0
    return target, mask


# ------------------------------------------------------------------
# Dataset / DataLoader helpers
# ------------------------------------------------------------------

class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: List[List[Tuple[int, int]]], num_kcs: int):
        self.sequences = sequences
        self.num_kcs = num_kcs

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        seq = self.sequences[idx]
        x = encode_sequence(seq, self.num_kcs)
        target, mask = encode_target(seq, self.num_kcs)
        return x, target, mask, len(seq)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, targets, masks, lengths = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True)
    t_padded = pad_sequence(targets, batch_first=True)
    m_padded = pad_sequence(masks, batch_first=True)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    return x_padded, t_padded, m_padded, lengths_t


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_dkt(
    model: DKTModel,
    train_sequences: List[List[Tuple[int, int]]],
    *,
    num_kcs: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu",
    val_sequences: Optional[List[List[Tuple[int, int]]]] = None,
    patience: int = 5,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train DKT model.

    Returns:
        {"train_loss": [...], "val_auc": [...]}
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")

    dataset = DKTDataset(train_sequences, num_kcs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_auc": []}
    best_val_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x_batch, t_batch, m_batch, lens in loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            m_batch = m_batch.to(device)

            pred, _ = model(x_batch, lens)
            loss_matrix = criterion(pred, t_batch) * m_batch
            loss = loss_matrix.sum() / (m_batch.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * m_batch.sum().item()
            total_count += m_batch.sum().item()

        avg_loss = total_loss / (total_count + 1e-8)
        history["train_loss"].append(avg_loss)

        # Validation
        val_auc = 0.0
        if val_sequences:
            val_auc = evaluate_dkt_auc(model, val_sequences, num_kcs, device)
            history["val_auc"].append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch + 1}")
                    break

        if verbose and (epoch + 1) % 5 == 0:
            val_str = f"  val_auc={val_auc:.4f}" if val_sequences else ""
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}{val_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def evaluate_dkt_auc(
    model: DKTModel,
    sequences: List[List[Tuple[int, int]]],
    num_kcs: int,
    device: str = "cpu",
) -> float:
    """Compute AUC on validation sequences."""
    model.eval()
    all_preds: List[float] = []
    all_labels: List[int] = []

    dataset = DKTDataset(sequences, num_kcs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=collate_fn,
    )

    with torch.no_grad():
        for x_batch, t_batch, m_batch, lens in loader:
            x_batch = x_batch.to(device)
            pred, _ = model(x_batch, lens)
            pred = pred.cpu()

            for i in range(pred.size(0)):
                for t in range(lens[i].item() - 1):
                    kc_mask = m_batch[i, t]
                    if kc_mask.sum() == 0:
                        continue
                    kc_idx = kc_mask.nonzero(as_tuple=True)[0]
                    for k in kc_idx:
                        all_preds.append(pred[i, t, k].item())
                        all_labels.append(int(t_batch[i, t, k].item()))

    if len(all_preds) < 2 or len(set(all_labels)) < 2:
        return 0.5

    # Simple AUC computation
    return _auc(all_labels, all_preds)


def _auc(labels: List[int], preds: List[float]) -> float:
    """Compute AUC via sorting."""
    pairs = sorted(zip(preds, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


# ------------------------------------------------------------------
# DKT Predictor (inference)
# ------------------------------------------------------------------

class DKTPredictor:
    """Inference wrapper for trained DKT model."""

    def __init__(
        self,
        model: DKTModel,
        kc_to_idx: Dict[str, int],
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.kc_to_idx = kc_to_idx
        self.num_kcs = model.num_kcs
        self.device = device
        self._hidden: Optional[torch.Tensor] = None
        self._last_pred: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._hidden = None
        self._last_pred = None

    def process_sequence(self, seq: List[Tuple[str, int]]) -> None:
        """Process a sequence of (kc_name, correct) observations."""
        if not seq:
            return
        idx_seq = []
        for kc, correct in seq:
            kc_idx = self.kc_to_idx.get(kc)
            if kc_idx is None:
                continue
            idx_seq.append((kc_idx, correct))
        if not idx_seq:
            return
        x = encode_sequence(idx_seq, self.num_kcs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred, self._hidden = self.model(x, hidden=self._hidden)
        self._last_pred = pred[0, -1]  # (num_kcs,) — prediction after last input

    def predict(self, kc: str) -> float:
        """P(correct) for a question on this KC, based on processed history."""
        if self._last_pred is None:
            return 0.5  # no history yet, return uniform
        kc_idx = self.kc_to_idx.get(kc)
        if kc_idx is None:
            return 0.5
        return self._last_pred[kc_idx].item()


# ------------------------------------------------------------------
# Save / Load
# ------------------------------------------------------------------

def save_dkt_checkpoint(
    model: DKTModel,
    kc_to_idx: Dict[str, int],
    path: str,
    history: Optional[Dict] = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "num_kcs": model.num_kcs,
        "hidden_dim": model.hidden_dim,
        "kc_to_idx": kc_to_idx,
    }
    if history:
        checkpoint["history"] = history
    torch.save(checkpoint, path)


def load_dkt_checkpoint(
    path: str, device: str = "cpu",
) -> Tuple[DKTModel, Dict[str, int]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DKTModel(
        num_kcs=checkpoint["num_kcs"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["kc_to_idx"]
