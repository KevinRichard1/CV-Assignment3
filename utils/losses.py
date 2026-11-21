"""Classification losses for one-shot training with optional label mixing."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor | None = None,
    lam: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy for ordinary examples, MixUp, or CutMix.

    For MixUp/CutMix, compute the usual convex combination of the two
    cross-entropy losses. No energy interpretation or margin term is used.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes]")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")

    loss_a = F.cross_entropy(logits, labels_a)
    if labels_b is None or lam >= 1.0:
        return loss_a
    loss_b = F.cross_entropy(logits, labels_b)
    return lam * loss_a + (1.0 - lam) * loss_b
