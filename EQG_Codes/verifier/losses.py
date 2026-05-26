import torch
import torch.nn.functional as F



def info_nce_loss(logits, labels, tau=0.07):
    """
    h_x:  [B, D]
    h_cd: [B, D]
    return: loss, acc
    """

    loss = F.cross_entropy(logits, labels)

    # accuracy: whether the positive index is the row-wise argmax
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()

    return loss, acc