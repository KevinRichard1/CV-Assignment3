"""Conventional MLP classifier for frozen SigLIP image embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Map a SigLIP image embedding to one class logit per target class.

    The outputs are ordinary classification logits. Higher logits indicate
    greater compatibility with a class; train with cross-entropy loss.
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
            raise ValueError(
                "embedding_dim and hidden_dim must be positive; num_classes must be >= 2"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)

        self.norm = nn.LayerNorm(self.embedding_dim)
        self.hidden = nn.Linear(self.embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, num_classes]."""
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings [B, {self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        x = self.norm(embeddings)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.head(x)

    def probabilities(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Convert logits to probabilities; higher logits mean higher probability."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        return torch.softmax(self(embeddings) / temperature, dim=-1)

    @staticmethod
    def predict_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def freeze_except_head(self) -> None:
        """For optional TTA, update only the final classification layer."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
