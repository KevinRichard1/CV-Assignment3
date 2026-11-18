"""Loss functions for discriminative class-energy learning."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def energy_margin_loss(
    energies: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor | None = None,
    lam: float = 1.0,
    *,
    temperature: float = 1.0,
    margin: float = 0.5,
    margin_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CE on negative energies plus a correct-vs-nearest-negative margin.

    For MixUp/CutMix, CE uses the soft target. The margin term treats the
    convex combination of the two target-class energies as positive energy.
    """
    if energies.ndim != 2:
        raise ValueError("energies must have shape [batch, num_classes]")
    if temperature <= 0 or margin_weight < 0:
        raise ValueError("temperature must be > 0 and margin_weight >= 0")
    if labels_b is None:
        labels_b = labels_a
    logits = -energies / temperature
    ce_a = F.cross_entropy(logits, labels_a)
    ce_b = F.cross_entropy(logits, labels_b)
    ce = lam * ce_a + (1.0 - lam) * ce_b

    target_energy = lam * energies.gather(1, labels_a[:, None]).squeeze(1)
    target_energy = target_energy + (1.0 - lam) * energies.gather(
        1, labels_b[:, None]
    ).squeeze(1)

    negative_energies = energies.clone()
    negative_energies.scatter_(1, labels_a[:, None], float("inf"))
    negative_energies.scatter_(1, labels_b[:, None], float("inf"))
    nearest_negative = negative_energies.min(dim=1).values
    margin_loss = F.relu(margin + target_energy - nearest_negative).mean()

    total = ce + margin_weight * margin_loss
    return total, {"ce": ce.detach(), "margin": margin_loss.detach()}
