"""Train a one-shot SigLIP + conventional MLP classifier.

Expected training layout:
    data/train/<class_name>/*.{jpg,png,...}

Exactly one original source image is selected per class. All other source
images under data/train are ignored. Augmented views of the selected 16 images
are generated dynamically during training. No validation or test data is read.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.classifier import MLPClassifier
from utils.augment import apply_cutout, apply_mixup_cutmix, build_train_transform
from utils.losses import classification_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/oneshot_siglip"))
    parser.add_argument("--encoder-name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cutout-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_pil(batch: list[tuple[Any, int]]) -> tuple[list[Any], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def select_exactly_one_per_class(
    dataset: ImageFolder,
    *,
    seed: int,
    expected_classes: int,
) -> tuple[list[int], list[dict[str, str]]]:
    """Select one and only one original file per class, reproducibly."""
    if len(dataset.classes) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} class folders, found {len(dataset.classes)}: "
            f"{dataset.classes}"
        )

    indices_by_class: dict[int, list[int]] = {
        class_index: [] for class_index in range(len(dataset.classes))
    }
    for index, (_, class_index) in enumerate(dataset.samples):
        indices_by_class[class_index].append(index)

    rng = random.Random(seed)
    selected_indices: list[int] = []
    manifest: list[dict[str, str]] = []
    for class_index, class_name in enumerate(dataset.classes):
        candidates = indices_by_class[class_index]
        if not candidates:
            raise ValueError(f"Class {class_name!r} contains no supported images.")
        selected_index = rng.choice(candidates)
        selected_indices.append(selected_index)
        source_path, _ = dataset.samples[selected_index]
        manifest.append(
            {
                "class_name": class_name,
                "class_index": str(class_index),
                "image": str(Path(source_path).resolve()),
            }
        )
    return selected_indices, manifest


def save_checkpoint(
    path: Path,
    *,
    classifier: MLPClassifier,
    encoder_name: str,
    classes: list[str],
    class_to_idx: dict[str, int],
    selected_samples: list[dict[str, str]],
    args: argparse.Namespace,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "encoder_name": encoder_name,
            "embedding_dim": classifier.embedding_dim,
            "num_classes": classifier.num_classes,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "selected_samples": selected_samples,
            "epoch": epoch,
            "train_args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.num_classes < 2:
        raise ValueError("num-classes must be at least 2")
    set_seed(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # The transform is applied only after a single source image per class has
    # been selected. No other file in a class folder enters the training loop.
    train_dataset = ImageFolder(
        args.train_dir,
        transform=build_train_transform(args.image_size),
    )
    selected_indices, selected_manifest = select_exactly_one_per_class(
        train_dataset,
        seed=args.selection_seed,
        expected_classes=args.num_classes,
    )
    selected_dataset = Subset(train_dataset, selected_indices)

    manifest_path = args.output_dir / "selected_one_per_class.json"
    manifest_path.write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")
    print(f"Selected exactly {len(selected_indices)} original images ({manifest_path})")
    for row in selected_manifest:
        print(f"  {row['class_name']}: {Path(row['image']).name}")

    train_loader = DataLoader(
        selected_dataset,
        batch_size=min(args.batch_size, len(selected_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pil,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    from models.encoder import SigLIPEncoder

    encoder = SigLIPEncoder(
        args.encoder_name,
        local_files_only=args.local_files_only,
    ).to(device)
    encoder.eval()
    classifier = MLPClassifier(
        embedding_dim=encoder.embedding_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    for epoch in range(1, args.epochs + 1):
        encoder.eval()
        classifier.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
        )
        for step, (images, labels) in enumerate(progress, start=1):
            labels = labels.to(device, non_blocking=True)
            pixel_values = encoder.preprocess(images).to(device, non_blocking=True)
            pixel_values = apply_cutout(pixel_values, probability=args.cutout_prob)
            pixel_values, labels_a, labels_b, lam = apply_mixup_cutmix(
                pixel_values,
                labels,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
            )

            # SigLIP remains frozen; only the MLP classifier receives gradients.
            with torch.no_grad():
                embeddings = encoder(pixel_values)
            logits = classifier(embeddings)
            loss = classification_loss(logits, labels_a, labels_b, lam)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
            optimizer.step()

            current_batch = labels.shape[0]
            running_loss += float(loss.detach().item()) * current_batch
            seen += current_batch
            if args.log_every > 0 and step % args.log_every == 0:
                progress.set_postfix(
                    loss=f"{running_loss / max(seen, 1):.4f}",
                )

        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"loss={running_loss / max(seen, 1):.5f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    # No validation split is assumed. Save the final fixed-epoch model.
    final_path = args.output_dir / "final.pt"
    save_checkpoint(
        final_path,
        classifier=classifier,
        encoder_name=args.encoder_name,
        classes=train_dataset.classes,
        class_to_idx=train_dataset.class_to_idx,
        selected_samples=selected_manifest,
        args=args,
        epoch=args.epochs,
    )
    print(f"Final checkpoint: {final_path}")
    print(f"Selected-example manifest: {manifest_path}")


if __name__ == "__main__":
    main()
