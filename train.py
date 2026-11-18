"""Train a one-shot SigLIP + discriminative energy classifier.

Expected training layout:
    data/train/<class_name>/*.{jpg,png,...}
Optional validation layout:
    data/val/<class_name>/*.{jpg,png,...}

Exactly one source image is selected per class. Augmented views are generated
on the fly; the other source images in data/train are never loaded for training.
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

from models.classifier import EnergyClassifier
from models.encoder import SigLIPEncoder
from utils.augment import apply_cutout, apply_mixup_cutmix, build_train_transform
from utils.losses import energy_margin_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/oneshot_siglip"))
    parser.add_argument("--encoder-name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cutout-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
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
    indices_by_class: dict[int, list[int]] = {i: [] for i in range(len(dataset.classes))}
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


@torch.no_grad()
def extract_embeddings(
    encoder: SigLIPEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    encoder.eval()
    for images, labels in tqdm(loader, desc="Encoding validation", leave=False):
        pixel_values = encoder.preprocess(images).to(device, non_blocking=True)
        all_embeddings.append(encoder(pixel_values).cpu())
        all_labels.append(labels.cpu())
    if not all_embeddings:
        raise ValueError("Validation directory contains no images.")
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def evaluate_cached(
    classifier: EnergyClassifier,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    classifier.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], 512):
            batch = embeddings[start : start + 512].to(device)
            predictions.append(classifier(batch).argmin(dim=-1).cpu())
    predicted = torch.cat(predictions)
    return float(predicted.eq(labels).float().mean().item())


def save_checkpoint(
    path: Path,
    *,
    classifier: EnergyClassifier,
    encoder_name: str,
    classes: list[str],
    class_to_idx: dict[str, int],
    selected_samples: list[dict[str, str]],
    args: argparse.Namespace,
    epoch: int,
    val_accuracy: float | None,
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
            "val_accuracy": val_accuracy,
            "train_args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

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
    # Persist the exact originals used. No other file under train-dir is loaded.
    manifest_path = args.output_dir / "selected_one_per_class.json"
    manifest_path.write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")
    print(f"Selected exactly {len(selected_indices)} originals ({manifest_path})")
    for row in selected_manifest:
        print(f"  {row['class_name']}: {Path(row['image']).name}")

    train_loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pil,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    encoder = SigLIPEncoder(
        args.encoder_name,
        local_files_only=args.local_files_only,
    ).to(device)
    encoder.eval()
    classifier = EnergyClassifier(
        embedding_dim=encoder.embedding_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    val_embeddings: torch.Tensor | None = None
    val_labels: torch.Tensor | None = None
    if args.val_dir is not None:
        val_dataset = ImageFolder(args.val_dir, transform=None)
        if val_dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError(
                "Validation class folders must match training class folders exactly.\n"
                f"Train: {train_dataset.class_to_idx}\nVal: {val_dataset.class_to_idx}"
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_pil,
            persistent_workers=args.num_workers > 0,
        )
        val_embeddings, val_labels = extract_embeddings(encoder, val_loader, device)
        print(f"Cached {len(val_labels)} validation embeddings.")

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        encoder.eval()
        classifier.train()
        running_loss = 0.0
        running_ce = 0.0
        running_margin = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, (images, labels) in enumerate(progress, start=1):
            labels = labels.to(device, non_blocking=True)
            pixel_values = encoder.preprocess(images).to(device, non_blocking=True)
            pixel_values = apply_cutout(
                pixel_values, probability=args.cutout_prob
            )
            pixel_values, labels_a, labels_b, lam = apply_mixup_cutmix(
                pixel_values,
                labels,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
            )

            with torch.no_grad():
                embeddings = encoder(pixel_values)
            energies = classifier(embeddings)
            loss, loss_parts = energy_margin_loss(
                energies,
                labels_a,
                labels_b,
                lam,
                temperature=args.temperature,
                margin=args.margin,
                margin_weight=args.margin_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = labels.shape[0]
            running_loss += float(loss.detach().item()) * batch_size
            running_ce += float(loss_parts["ce"].item()) * batch_size
            running_margin += float(loss_parts["margin"].item()) * batch_size
            seen += batch_size
            if args.log_every > 0 and step % args.log_every == 0:
                progress.set_postfix(
                    loss=f"{running_loss / max(seen, 1):.4f}",
                    ce=f"{running_ce / max(seen, 1):.4f}",
                    margin=f"{running_margin / max(seen, 1):.4f}",
                )

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_accuracy = None
        if val_embeddings is not None and val_labels is not None:
            val_accuracy = evaluate_cached(classifier, val_embeddings, val_labels, device)
            print(
                f"Epoch {epoch:03d}: loss={train_loss:.5f}, "
                f"val_acc={val_accuracy:.4%}, lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(
                    args.output_dir / "best.pt",
                    classifier=classifier,
                    encoder_name=args.encoder_name,
                    classes=train_dataset.classes,
                    class_to_idx=train_dataset.class_to_idx,
                    selected_samples=selected_manifest,
                    args=args,
                    epoch=epoch,
                    val_accuracy=val_accuracy,
                )
        else:
            print(
                f"Epoch {epoch:03d}: loss={train_loss:.5f}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    save_checkpoint(
        args.output_dir / "last.pt",
        classifier=classifier,
        encoder_name=args.encoder_name,
        classes=train_dataset.classes,
        class_to_idx=train_dataset.class_to_idx,
        selected_samples=selected_manifest,
        args=args,
        epoch=args.epochs,
        val_accuracy=best_accuracy if best_accuracy >= 0 else None,
    )
    if best_accuracy >= 0:
        print(f"Best validation accuracy: {best_accuracy:.4%}")
        print(f"Best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"Last checkpoint: {args.output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
