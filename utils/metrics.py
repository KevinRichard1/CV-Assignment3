"""Small dependency-free classification metrics."""
from __future__ import annotations

import torch


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, object]:
    predictions = predictions.detach().cpu().long().view(-1)
    targets = targets.detach().cpu().long().view(-1)
    if predictions.numel() != targets.numel():
        raise ValueError("predictions and targets must have the same length")
    correct = predictions.eq(targets)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            confusion[target, prediction] += 1
    per_class_total = confusion.sum(dim=1)
    per_class_correct = confusion.diag()
    per_class_accuracy = torch.where(
        per_class_total > 0,
        per_class_correct.float() / per_class_total.clamp_min(1),
        torch.full_like(per_class_total, float("nan"), dtype=torch.float32),
    )
    return {
        "accuracy": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": confusion.tolist(),
        "count": int(targets.numel()),
    }
