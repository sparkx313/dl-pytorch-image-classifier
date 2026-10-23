"""Train a small image classifier from labeled folders and classify real images."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]], image_size: int):
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.records[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                pixels = np.asarray(image, dtype=np.float32) / 255.0
        except OSError as exc:
            raise ValueError(f"Could not read image {path}: {exc}") from exc
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        return tensor, label


class SmallCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(192, class_count))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def discover_images(root: Path) -> tuple[list[str], list[tuple[Path, int]]]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Dataset needs at least two class subdirectories.")
    classes = [path.name for path in class_dirs]
    records: list[tuple[Path, int]] = []
    for label, directory in enumerate(class_dirs):
        images = sorted(
            path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"Class directory {directory.name!r} contains no supported images.")
        records.extend((path, label) for path in images)
    return classes, records


def split_records(
    records: list[tuple[Path, int]], validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = np.random.default_rng(seed)
    labels = sorted({label for _path, label in records})
    train: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    for label in labels:
        class_records = [record for record in records if record[1] == label]
        if len(class_records) < 2:
            raise ValueError(
                f"Every class needs at least two images for a train/validation split (class {label})."
            )
        order = rng.permutation(len(class_records))
        validation_count = max(1, int(round(len(class_records) * validation_fraction)))
        validation_count = min(validation_count, len(class_records) - 1)
        validation.extend(class_records[index] for index in order[:validation_count])
        train.extend(class_records[index] for index in order[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot see a CUDA device.")
    return torch.device(requested)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int) -> tuple[float, float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    predictions: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(loss_function(logits, labels).item())
            total_samples += len(labels)
            predictions.append(logits.argmax(dim=1).cpu())
            truths.append(labels.cpu())
    if not total_samples:
        raise ValueError("Validation split contains no images.")
    predicted = torch.cat(predictions).numpy()
    truth = torch.cat(truths).numpy()
    f1_scores: list[float] = []
    for label in range(class_count):
        true_positive = int(np.sum((truth == label) & (predicted == label)))
        false_positive = int(np.sum((truth != label) & (predicted == label)))
        false_negative = int(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    return total_loss / total_samples, float(np.mean(predicted == truth)), float(np.mean(f1_scores))


def train(args: argparse.Namespace) -> None:
    if args.image_size < 32:
        raise ValueError("Image size must be at least 32 pixels for this CNN.")
    classes, records = discover_images(args.data_dir)
    train_records, validation_records = split_records(records, args.validation_fraction, args.seed)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ImageFolderDataset(train_records, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        ImageFolderDataset(validation_records, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = SmallCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    print(
        f"Training on {len(train_records)} images; validating on {len(validation_records)} "
        f"images across {len(classes)} classes using {device}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(labels)
            train_count += len(labels)

        validation_loss, validation_accuracy, validation_f1 = evaluate(
            model, validation_loader, device, len(classes)
        )
        train_loss = train_loss_total / train_count
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.5f} val_loss={validation_loss:.5f} "
            f"val_accuracy={validation_accuracy:.4f} val_macro_f1={validation_f1:.4f}"
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_weights is None:
        raise ValueError("Training did not produce a finite validation checkpoint.")
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_weights,
            "class_names": classes,
            "image_size": args.image_size,
        },
        args.model_out,
    )
    print(f"Saved best validation checkpoint from epoch {best_epoch} to {args.model_out}.")


def load_checkpoint(path: Path, device: torch.device) -> tuple[SmallCNN, list[str], int]:
    if not path.is_file():
        raise ValueError(f"Model checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError, ValueError) as exc:
        raise ValueError(f"Could not safely load model checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("class_names"), list):
        raise ValueError("Checkpoint is missing supported class metadata.")
    classes = checkpoint["class_names"]
    image_size = checkpoint.get("image_size")
    state_dict = checkpoint.get("state_dict")
    if (
        not classes
        or not all(isinstance(name, str) for name in classes)
        or not isinstance(image_size, int)
        or image_size < 32
        or not isinstance(state_dict, dict)
    ):
        raise ValueError("Checkpoint contains invalid model metadata.")
    model = SmallCNN(len(classes))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match the classifier architecture: {exc}") from exc
    model.to(device)
    model.eval()
    return model, classes, image_size


def predict(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, classes, image_size = load_checkpoint(args.model, device)
    try:
        with Image.open(args.image) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except OSError as exc:
        raise ValueError(f"Could not read image {args.image}: {exc}") from exc
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = torch.softmax(model(tensor)[0], dim=0)
    count = min(args.top_k, len(classes))
    values, indices = torch.topk(scores, count)
    results = [
        {"class": classes[int(index)], "confidence": round(float(value), 6)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    print(json.dumps({"image": str(args.image), "predictions": results}, indent=2))


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Validation fraction must be between 0 and 1.") from exc
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("Validation fraction must be between 0 and 1.")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive finite number.") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive finite number.")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.") from exc
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small CNN on user-provided labeled image folders.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and evaluate a classifier from a folder dataset.")
    training.add_argument("--data-dir", required=True, type=Path, help="Root directory with one subfolder per class.")
    training.add_argument("--model-out", type=Path, default=Path("image_classifier.pt"))
    training.add_argument("--epochs", type=positive_int, default=30)
    training.add_argument("--patience", type=positive_int, default=6)
    training.add_argument("--batch-size", type=positive_int, default=32)
    training.add_argument("--image-size", type=positive_int, default=128)
    training.add_argument("--learning-rate", type=positive_float, default=0.001)
    training.add_argument("--weight-decay", type=nonnegative_float, default=0.0001)
    training.add_argument("--validation-fraction", type=fraction, default=0.2)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    inference = commands.add_parser("predict", help="Classify one image with a saved checkpoint.")
    inference.add_argument("--image", required=True, type=Path)
    inference.add_argument("--model", required=True, type=Path)
    inference.add_argument("--top-k", type=positive_int, default=3)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "train":
            if args.model_out.resolve() == args.data_dir.resolve():
                raise ValueError("Model output must not overwrite the dataset directory.")
            train(args)
        else:
            predict(args)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
