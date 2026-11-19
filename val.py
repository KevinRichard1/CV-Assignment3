"""Evaluate or predict with a saved one-shot SigLIP energy classifier.

Labelled layout (reports accuracy):
    data/test/<class_name>/*.{jpg,png,...}

Unlabelled layout (writes predictions, no accuracy is computed):
    data/test/*.{jpg,png,...}  # images may also be nested in subfolders

Modes: standard, multiview, adapt. In adapt mode, only a temporary copy of the
classifier's final energy layer is updated on each batch. When data-dir contains
class-named subfolders matching the training folders, labels are used only to
report evaluation metrics; they are never passed to the adaptation objective.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.adaptation import EpisodicEnergyTTA
from models.classifier import EnergyClassifier
from utils.augment import build_tta_transforms
from utils.metrics import classification_metrics

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class UnlabelledImageFolder(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No supported images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, None, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            return image.convert("RGB"), None, str(path)


class LabelledImageFolder(ImageFolder):
    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        image, label = super().__getitem__(index)
        return image, label, self.samples[index][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("standard", "single", "multiview", "adapt"),
        default="standard",
        help="standard/single: one view; multiview: average views; adapt: episodic unsupervised TTA",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--views", type=int, default=4, help="Total views, including original")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tta-steps", type=int, default=5)
    parser.add_argument("--tta-lr", type=float, default=1e-4)
    parser.add_argument("--tta-consistency-weight", type=float, default=0.25)
    parser.add_argument("--tta-anchor-weight", type=float, default=1.0)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_images(batch: list[tuple[Image.Image, int | None, str]]) -> tuple[list[Image.Image], torch.Tensor | None, list[str]]:
    images, labels, paths = zip(*batch)
    targets = None if labels[0] is None else torch.tensor(labels, dtype=torch.long)
    return list(images), targets, list(paths)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # Checkpoints are produced locally by train.py and include model metadata.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {"classifier_state_dict", "encoder_name", "embedding_dim", "num_classes", "classes"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    return checkpoint


def make_dataset(data_dir: Path, class_names: list[str]) -> tuple[Dataset, bool]:
    # Only treat it as labelled when every checkpoint class has a matching folder.
    has_class_folders = all((data_dir / class_name).is_dir() for class_name in class_names)
    if has_class_folders:
        dataset = LabelledImageFolder(data_dir, transform=None)
        if dataset.classes != class_names:
            raise ValueError(
                f"Evaluation class ordering differs from checkpoint.\n"
                f"Checkpoint: {class_names}\nEvaluation: {dataset.classes}"
            )
        return dataset, True
    return UnlabelledImageFolder(data_dir), False


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.temperature <= 0:
        raise ValueError("batch-size must be positive and temperature must be > 0")
    if args.mode == "single":
        args.mode = "standard"
    if args.mode != "standard" and args.views < 2:
        raise ValueError("multiview/adapt mode requires --views >= 2")

    device = choose_device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    classes = list(checkpoint["classes"])
    dataset, is_labelled = make_dataset(args.data_dir, classes)
    if is_labelled and dataset.class_to_idx != checkpoint.get("class_to_idx", dataset.class_to_idx):
        raise ValueError("Evaluation class indices do not match the checkpoint.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_images,
        persistent_workers=args.num_workers > 0,
    )

    from models.encoder import SigLIPEncoder

    encoder = SigLIPEncoder(
        checkpoint["encoder_name"], local_files_only=args.local_files_only
    ).to(device)
    classifier = EnergyClassifier(
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        hidden_dim=int(checkpoint["train_args"].get("hidden_dim", 512)),
        dropout=float(checkpoint["train_args"].get("dropout", 0.1)),
    ).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()
    encoder.eval()

    if args.mode == "standard":
        view_transforms = [None]
    else:
        image_size = int(checkpoint["train_args"].get("image_size", 224))
        view_transforms = build_tta_transforms(image_size=image_size, views=args.views)

    adapter = EpisodicEnergyTTA(
        steps=args.tta_steps,
        learning_rate=args.tta_lr,
        temperature=args.temperature,
        consistency_weight=args.tta_consistency_weight,
        anchor_weight=args.tta_anchor_weight,
    )

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    csv_rows: list[dict[str, object]] = []

    for images, labels, paths in tqdm(loader, desc=f"Predicting ({args.mode})"):
        batch_size = len(images)
        all_view_pixels: list[torch.Tensor] = []
        for transform in view_transforms:
            view_images = [image if transform is None else transform(image) for image in images]
            all_view_pixels.append(encoder.preprocess(view_images))

        # One encoder invocation per batch; all views are concatenated along B.
        pixel_values = torch.cat(all_view_pixels, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            features = encoder(pixel_values)
        view_embeddings = features.reshape(len(view_transforms), batch_size, -1)

        if args.mode == "adapt":
            probabilities, _ = adapter.adapt_and_predict(classifier, view_embeddings)
        else:
            with torch.no_grad():
                per_view_probabilities = torch.stack(
                    [
                        classifier.probabilities(view_embeddings[v], args.temperature)
                        for v in range(len(view_transforms))
                    ],
                    dim=0,
                )
                probabilities = per_view_probabilities.mean(dim=0)

        predictions = probabilities.argmax(dim=-1).cpu()
        confidences = probabilities.max(dim=-1).values.cpu()
        if labels is not None:
            all_predictions.append(predictions)
            all_targets.append(labels.cpu())

        for offset, path in enumerate(paths):
            predicted_index = int(predictions[offset].item())
            row: dict[str, object] = {
                "image": path,
                "predicted_index": predicted_index,
                "predicted_class": classes[predicted_index],
                "confidence": float(confidences[offset].item()),
            }
            if labels is not None:
                target_index = int(labels[offset].item())
                row["target_index"] = target_index
                row["target_class"] = classes[target_index]
            csv_rows.append(row)

    if is_labelled:
        predictions_tensor = torch.cat(all_predictions)
        targets_tensor = torch.cat(all_targets)
        metrics = classification_metrics(predictions_tensor, targets_tensor, len(classes))
        print(f"Evaluation mode: {args.mode}")
        print(f"Labeled test images: {len(targets_tensor)}")
        print(f"Top-1 accuracy: {metrics['accuracy']:.4%} ({metrics['count']} images)")
        print("Per-class accuracy:")
        for class_name, accuracy in zip(classes, metrics["per_class_accuracy"]):
            value = "n/a" if accuracy != accuracy else f"{accuracy:.4%}"
            print(f"  {class_name}: {value}")
    else:
        print(f"Predicted {len(csv_rows)} images across {len(classes)} classes (no matching class labels found).")
        if args.predictions_csv is None:
            args.predictions_csv = Path("predictions.csv")

    if args.predictions_csv is not None:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image", "predicted_index", "predicted_class", "confidence"]
        if is_labelled:
            fieldnames += ["target_index", "target_class"]
        with args.predictions_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Predictions written to: {args.predictions_csv}")


if __name__ == "__main__":
    main()
