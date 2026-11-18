"""Frozen Hugging Face SigLIP image encoder."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIPEncoder(nn.Module):
    """Load a pretrained SigLIP model and expose normalized image embeddings.

    The encoder is intentionally frozen and kept in eval mode, even when a
    parent module is switched to train mode. `preprocess` accepts PIL images;
    `forward` accepts pixel_values returned by the Hugging Face processor.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        *,
        local_files_only: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.requires_grad_(False)
        self.model.eval()

        config = self.model.config
        projection_dim = getattr(config, "projection_dim", None)
        if projection_dim is None and hasattr(config, "vision_config"):
            projection_dim = getattr(config.vision_config, "hidden_size", None)
        if projection_dim is None:
            projection_dim = getattr(config, "hidden_size", None)
        if projection_dim is None:
            raise ValueError("Could not infer SigLIP embedding dimension from model config.")
        self.embedding_dim = int(projection_dim)

    def train(self, mode: bool = True) -> "SigLIPEncoder":
        # The feature extractor must never enter train mode.
        super().train(False)
        self.model.eval()
        return self

    def preprocess(self, images: Sequence[Image.Image] | Image.Image) -> torch.Tensor:
        """Run the official SigLIP image processor and return pixel_values."""
        if isinstance(images, Image.Image):
            images = [images]
        batch = self.processor(images=list(images), return_tensors="pt")
        if "pixel_values" not in batch:
            raise RuntimeError("The Hugging Face processor did not return pixel_values.")
        return batch["pixel_values"]

    @staticmethod
    def _pool_features(output: object) -> torch.Tensor:
        """Handle Tensor and ModelOutput variants across Transformers versions."""
        if torch.is_tensor(output):
            features = output
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            features = output.pooler_output
        elif hasattr(output, "image_embeds") and output.image_embeds is not None:
            features = output.image_embeds
        elif isinstance(output, (tuple, list)) and output:
            first = output[0]
            if torch.is_tensor(first) and first.ndim == 2:
                features = first
            elif hasattr(output, "pooler_output"):
                features = output.pooler_output
            else:
                raise RuntimeError(
                    "SigLIP returned sequence features without a pooled image embedding."
                )
        else:
            raise TypeError(f"Unsupported SigLIP feature output: {type(output)!r}")

        if features.ndim != 2:
            raise RuntimeError(
                f"Expected [batch, embedding_dim] pooled features, got {tuple(features.shape)}."
            )
        return features

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"pixel_values must have shape [B, C, H, W], got {tuple(pixel_values.shape)}"
            )
        self.model.eval()
        output = self.model.get_image_features(pixel_values=pixel_values)
        features = self._pool_features(output).float()
        if self.normalize:
            features = F.normalize(features, p=2, dim=-1)
        return features

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[Image.Image] | Image.Image,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Convenience API for PIL input, including preprocessing."""
        pixel_values = self.preprocess(images)
        if device is not None:
            pixel_values = pixel_values.to(device)
        return self(pixel_values)
