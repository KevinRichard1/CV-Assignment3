"""Small discriminative energy head for frozen SigLIP embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyClassifier(nn.Module):
    """Map an image embedding to one energy per class (lower is better).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 16,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes < 2 or hidden_dim <= 0:
            raise ValueError("embedding_dim/hidden_dim must be positive and num_classes >= 2")
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)

        self.norm = nn.LayerNorm(self.embedding_dim)
        self.hidden = nn.Linear(self.embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.energy_head = nn.Linear(hidden_dim, self.num_classes)

        # Small initial energies help keep early softmax gradients well-scaled.
        nn.init.normal_(self.energy_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.energy_head.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings [B, {self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        x = self.norm(embeddings)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.energy_head(x)

    def probabilities(self, embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        energies = self(embeddings)
        return torch.softmax(-energies / temperature, dim=-1)

    @staticmethod
    def predict_from_energies(energies: torch.Tensor) -> torch.Tensor:
        return energies.argmin(dim=-1)

    def freeze_except_energy_head(self) -> None:
        """For TTA, unfreeze only the final class-energy linear layer."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.energy_head.parameters():
            parameter.requires_grad_(True)
