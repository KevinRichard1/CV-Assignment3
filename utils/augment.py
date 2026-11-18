"""Image augmentations and optional batch-level mixing for one-shot training."""
from __future__ import annotations

import random
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms


def build_train_transform(image_size: int = 224) -> Callable:
    """Moderate, mostly label-preserving PIL augmentations."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.18),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.035),
        ]
    )


def build_tta_transforms(image_size: int = 224, views: int = 4) -> list[Callable | None]:
    """Return [identity, random crop views...]; total list length is `views`."""
    if views < 1:
        raise ValueError("views must be >= 1")
    result: list[Callable | None] = [None]
    crop_scales = [(0.82, 1.0), (0.72, 0.96), (0.88, 1.0), (0.76, 0.92)]
    while len(result) < views:
        scale = crop_scales[(len(result) - 1) % len(crop_scales)]
        result.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
    return result


def apply_cutout(
    pixel_values: torch.Tensor,
    *,
    probability: float = 0.0,
    scale: tuple[float, float] = (0.02, 0.12),
) -> torch.Tensor:
    """Apply a single zero-filled rectangle per selected normalized image.

    Zero in normalized pixel space corresponds approximately to the channel
    mean, making this a neutral cutout rather than an outlier-label operation.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not (0.0 < scale[0] <= scale[1] <= 1.0):
        raise ValueError("scale must satisfy 0 < min <= max <= 1")
    if probability == 0.0:
        return pixel_values

    x = pixel_values.clone()
    batch, _, height, width = x.shape
    for index in range(batch):
        if random.random() > probability:
            continue
        area_fraction = random.uniform(*scale)
        aspect = random.uniform(0.5, 2.0)
        cut_h = min(height, max(1, int((area_fraction * height * width / aspect) ** 0.5)))
        cut_w = min(width, max(1, int((area_fraction * height * width * aspect) ** 0.5)))
        top = random.randint(0, max(0, height - cut_h))
        left = random.randint(0, max(0, width - cut_w))
        x[index, :, top : top + cut_h, left : left + cut_w] = 0.0
    return x


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    center_x = random.randint(0, width - 1)
    center_y = random.randint(0, height - 1)
    x1 = max(0, center_x - cut_w // 2)
    x2 = min(width, center_x + cut_w // 2)
    y1 = max(0, center_y - cut_h // 2)
    y2 = min(height, center_y + cut_h // 2)
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    *,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Optionally mix a batch and return (images, y_a, y_b, lambda).

    MixUp/CutMix are applied to processor-normalized pixel tensors. Their
    labels are represented as a convex combination during loss computation.
    A batch size of one is left unchanged.
    """
    if mixup_alpha < 0 or cutmix_alpha < 0:
        raise ValueError("mixup_alpha and cutmix_alpha must be >= 0")
    labels_a = labels
    labels_b = labels
    lam = 1.0
    batch_size = pixel_values.shape[0]
    if batch_size < 2 or (mixup_alpha == 0 and cutmix_alpha == 0):
        return pixel_values, labels_a, labels_b, lam

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha == 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    distribution = torch.distributions.Beta(alpha, alpha)
    lam = float(distribution.sample().item())
    permutation = torch.randperm(batch_size, device=pixel_values.device)
    labels_b = labels[permutation]

    if use_cutmix:
        mixed = pixel_values.clone()
        _, _, height, width = mixed.shape
        x1, y1, x2, y2 = _rand_bbox(height, width, lam)
        mixed[:, :, y1:y2, x1:x2] = pixel_values[permutation, :, y1:y2, x1:x2]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        lam = 1.0 - box_area / float(height * width)
        pixel_values = mixed
    else:
        pixel_values = lam * pixel_values + (1.0 - lam) * pixel_values[permutation]

    return pixel_values, labels_a, labels_b, lam
