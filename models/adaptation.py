"""Optional episodic test-time adaptation of the energy head."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from .classifier import EnergyClassifier


class EpisodicEnergyTTA:
    """Adapt a temporary copy of the final energy layer on an unlabeled batch.

    The SigLIP encoder is not touched. Adaptation is reset for every batch so
    predictions do not depend on the order of earlier test batches.
    """

    def __init__(
        self,
        steps: int = 5,
        learning_rate: float = 1e-4,
        temperature: float = 1.0,
        consistency_weight: float = 0.25,
        anchor_weight: float = 1.0,
    ) -> None:
        if steps < 0 or learning_rate <= 0 or temperature <= 0:
            raise ValueError("steps must be >= 0 and learning_rate/temperature > 0")
        self.steps = steps
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.anchor_weight = anchor_weight

    @torch.enable_grad()
    def adapt_and_predict(
        self,
        classifier: EnergyClassifier,
        view_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, EnergyClassifier]:
        """Return mean per-view probabilities and the temporary adapted head.

        Args:
            classifier: trained classifier, left unchanged.
            view_embeddings: [num_views, batch, embedding_dim].
        """
        if view_embeddings.ndim != 3:
            raise ValueError("view_embeddings must have shape [views, batch, embedding_dim]")
        if view_embeddings.shape[0] < 1:
            raise ValueError("At least one view is required")

        base = classifier
        adapted = deepcopy(classifier)
        adapted.freeze_except_energy_head()
        # Keep dropout disabled during adaptation; gradients still flow to the
        # explicitly unfrozen final energy layer while behavior stays stable.
        adapted.eval()

        # Teacher probabilities from the unadapted checkpoint.
        with torch.no_grad():
            base_energies = torch.stack(
                [base(view_embeddings[v]) for v in range(view_embeddings.shape[0])], dim=0
            )
            teacher_probs = torch.softmax(-base_energies / self.temperature, dim=-1)

        if self.steps > 0:
            optimizer = torch.optim.AdamW(
                [p for p in adapted.parameters() if p.requires_grad],
                lr=self.learning_rate,
                weight_decay=1e-3,
            )
            for _ in range(self.steps):
                energies = torch.stack(
                    [adapted(view_embeddings[v]) for v in range(view_embeddings.shape[0])],
                    dim=0,
                )
                probabilities = torch.softmax(-energies / self.temperature, dim=-1)
                mean_probabilities = probabilities.mean(dim=0)

                # Per-image entropy (not entropy of the batch class histogram).
                entropy = -(
                    mean_probabilities * mean_probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()

                # Encourage different views of the same image to agree.
                consistency = (probabilities - mean_probabilities.unsqueeze(0)).square().mean()

                # Keep adaptation near the original model's predictions.
                log_probabilities = probabilities.clamp_min(1e-8).log()
                anchor = (
                    teacher_probs
                    * (teacher_probs.clamp_min(1e-8).log() - log_probabilities)
                ).sum(dim=-1).mean()

                loss = (
                    entropy
                    + self.consistency_weight * consistency
                    + self.anchor_weight * anchor
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        adapted.eval()
        with torch.no_grad():
            final_energies = torch.stack(
                [adapted(view_embeddings[v]) for v in range(view_embeddings.shape[0])], dim=0
            )
            final_probabilities = torch.softmax(
                -final_energies / self.temperature, dim=-1
            ).mean(dim=0)
        return final_probabilities, adapted
