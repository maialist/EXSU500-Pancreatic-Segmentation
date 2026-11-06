"""Train and run a patch-based 3D U-Net for pancreas CT segmentation."""
from __future__ import annotations

import argparse
from html import parser
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CLASS_IDS = (0, 1, 2)
CLASS_NAMES = {0: "background", 1: "pancreas", 2: "tumour"}
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_PATH = Path("data/models/unet3d.pt")
DEFAULT_PATCH_SIZE = (64, 64, 32)
NUM_DOWNSAMPLINGS = 3

## new - save checkpoints every epoch
def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    rng: np.random.Generator,
) -> None:
    """Save the complete state needed to resume training."""

    checkpoint = {
        "epoch": epoch,
        "best_dice": best_dice,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        # Random number generator states
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)

    print(f"Saved training checkpoint at epoch {epoch}", flush=True)
##

def target_labels(label: np.ndarray, target: str) -> np.ndarray:
    """Return labels for multiclass segmentation or tumour-vs-rest."""
    if target == "multiclass":
        return label
    if target == "tumour":
        return (label == 2).astype(np.uint8)
    raise ValueError(f"Unsupported target: {target}")


def _load_splits(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as split_file:
        splits = json.load(split_file)
    required = ("train", "val", "test")
    if not all(isinstance(splits.get(name), list) for name in required):
        raise ValueError(f"{path} must contain train, val, and test case lists")
    if any(not splits[name] for name in required):
        raise ValueError("Train, validation, and test splits must all be non-empty")
    if any(not isinstance(case, str) for name in required for case in splits[name]):
        raise ValueError("Split entries must be case-name strings")
    all_cases = splits["train"] + splits["val"] + splits["test"]
    if len(set(all_cases)) != len(all_cases):
        raise ValueError("Patient cases overlap between train, validation, and test")
    return splits


def _load_case(
    data_dir: Path, case_name: str, target: str
) -> tuple[np.ndarray, np.ndarray]:
    case_path = data_dir / f"{case_name}.npz"
    if not case_path.is_file():
        raise FileNotFoundError(f"Processed case is missing: {case_path}")
    with np.load(case_path) as case:
        image = case["image"].astype(np.float32, copy=False)
        raw_label = case["label"]
    if image.ndim != 3 or raw_label.ndim != 3:
        raise ValueError(f"Expected 3D image and label for {case_name}")
    if image.shape != raw_label.shape:
        raise ValueError(
            f"Image and label shapes differ for {case_name}: "
            f"{image.shape} != {raw_label.shape}"
        )
    if not np.isin(raw_label, CLASS_IDS).all():
        raise ValueError(f"Unexpected label value in processed case {case_name}")
    return image, target_labels(raw_label, target)


def _pad_case(
    image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    low_pad_d, low_pad_h, low_pad_w = (
        max(0, (size - dim) // 2) for dim, size in zip(image.shape, patch_size)
    )
    low_pad = (low_pad_d, low_pad_h, low_pad_w)
    high_pad = tuple(
        max(0, size - dim - low) for dim, size, low in zip(image.shape, patch_size, low_pad)
    )
    pad_width = tuple(zip(low_pad, high_pad))
    if any(low or high for low, high in pad_width):
        image = np.pad(image, pad_width, mode="constant")
        label = np.pad(label, pad_width, mode="constant")
    return image, label, (low_pad_d, low_pad_h, low_pad_w)


def _sample_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple[int, int, int],
    foreground_probability: float,
    rng: np.random.Generator,
    foreground_coordinates: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    image, label, low_pad = _pad_case(image, label, patch_size)
    foreground_classes = [class_id for class_id, coords in foreground_coordinates.items() if len(coords)]
    if foreground_classes and rng.random() < foreground_probability:
        class_id = int(rng.choice(foreground_classes))
        center = foreground_coordinates[class_id][rng.integers(len(foreground_coordinates[class_id]))]
        center = center + np.asarray(low_pad)
    else:
        center = np.asarray([rng.integers(dim) for dim in image.shape])

    start = np.maximum(0, center - np.asarray(patch_size) // 2)
    start = np.minimum(start, np.asarray(image.shape) - np.asarray(patch_size))
    slices = tuple(slice(int(begin), int(begin + size)) for begin, size in zip(start, patch_size))
    return image[slices], label[slices]


def _group_count(channels: int) -> int:
    return next(groups for groups in range(min(8, channels), 0, -1) if channels % groups == 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet3D(nn.Module):
    def __init__(self, out_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.enc1 = ConvBlock(1, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.bottleneck = ConvBlock(channels[2], channels[3])
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.up3 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(channels[0] * 2, channels[0])
        self.output = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_size = inputs.shape[2:]
        divisor = 2**NUM_DOWNSAMPLINGS
        padding = [(divisor - dim % divisor) % divisor for dim in original_size]
        inputs = F.pad(inputs, (0, padding[2], 0, padding[1], 0, padding[0]))

        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        center = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(torch.cat((self.up3(center), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))
        logits = self.output(dec1)
        return logits[
            :,
            :,
            : original_size[0],
            : original_size[1],
            : original_size[2],
        ]


def _dice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).movedim(-1, 1)
    probabilities = probabilities[:, 1:].flatten(start_dim=2)
    one_hot = one_hot[:, 1:].to(dtype=probabilities.dtype).flatten(start_dim=2)
    intersection = (probabilities * one_hot).sum(dim=2)
    denominator = probabilities.sum(dim=2) + one_hot.sum(dim=2)
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def _sliding_starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _predict_volume(
    model: UNet3D,
    image: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    padded_image, _, low_pad = _pad_case(
        image, np.zeros(image.shape, dtype=np.uint8), patch_size
    )
    spatial_shape = padded_image.shape
    locations = [
        (d, h, w)
        for d in _sliding_starts(spatial_shape[0], patch_size[0], overlap)
        for h in _sliding_starts(spatial_shape[1], patch_size[1], overlap)
        for w in _sliding_starts(spatial_shape[2], patch_size[2], overlap)
    ]
    probability_sum = np.zeros((model.output.out_channels, *spatial_shape), dtype=np.float32)
    prediction_count = np.zeros(spatial_shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(locations), batch_size):
            batch_locations = locations[batch_start : batch_start + batch_size]
            patches = np.stack(
                [
                    padded_image[
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ]
                    for d, h, w in batch_locations
                ]
            )
            inputs = torch.from_numpy(patches[:, None]).to(device=device)
            probabilities = torch.softmax(model(inputs), dim=1).cpu().numpy()
            for (d, h, w), patch_probabilities in zip(batch_locations, probabilities):
                slices = (
                    slice(d, d + patch_size[0]),
                    slice(h, h + patch_size[1]),
                    slice(w, w + patch_size[2]),
                )
                probability_sum[(slice(None), *slices)] += patch_probabilities
                prediction_count[slices] += 1

    probability_sum /= np.maximum(prediction_count, 1)[None]
    d0, h0, w0 = low_pad
    d1, h1, w1 = (d0 + dim for d0, dim in zip(low_pad, image.shape))
    prediction = probability_sum[:, d0:d1, h0:h1, w0:w1].argmax(axis=0).astype(np.uint8)
    return prediction


def _score_case(
    truth: np.ndarray, prediction: np.ndarray, target: str
) -> dict[int, float | None]:
    class_ids = (1,) if target == "tumour" else (1, 2)
    scores: dict[int, float | None] = {}
    for class_id in class_ids:
        truth_mask = truth == class_id
        prediction_mask = prediction == class_id
        denominator = int(truth_mask.sum() + prediction_mask.sum())
        scores[class_id] = (
            None
            if denominator == 0
            else 2.0
            * int(np.logical_and(truth_mask, prediction_mask).sum())
            / denominator
        )
    return scores


def _mean_foreground_dice(
    totals: dict[int, dict[str, float | int]]
) -> float:
    class_means = [
        float(values["dice_sum"]) / int(values["case_count"])
        for values in totals.values()
        if values["case_count"]
    ]
    return 0.0 if not class_means else float(np.mean(class_means))


def _validate(
    model: UNet3D,
    case_names: list[str],
    args: argparse.Namespace,
    target: str,
    device: torch.device,
) -> tuple[float, dict[int, dict[str, float | int]]]:
    class_ids = (1,) if target == "tumour" else (1, 2)
    totals = {class_id: {"dice_sum": 0.0, "case_count": 0} for class_id in class_ids}
    for case_index, case_name in enumerate(case_names, start=1):
        if args.verbose:
            print(
                f"  Validating case {case_index}/{len(case_names)}: {case_name}",
                flush=True,
            )

        image, truth = _load_case(args.data, case_name, target)

        if args.verbose:
            print(
                f"    Image shape: {image.shape}",
                flush=True,
            )

        prediction = _predict_volume(
            model,
            image,
            tuple(args.patch_size),
            args.overlap,
            args.inference_batch_size,
            device,
        )

        if args.verbose:
            print(
                f"    Prediction complete: {prediction.shape}",
                flush=True,
            )
        
        for class_id, score in _score_case(truth, prediction, target).items():
            if score is not None:
                totals[class_id]["dice_sum"] += score
                totals[class_id]["case_count"] += 1
    return _mean_foreground_dice(totals), totals


def _print_dice(totals: dict[int, dict[str, float | int]], target: str) -> None:
    for class_id, values in totals.items():
        name = "tumour" if target == "tumour" else CLASS_NAMES[class_id]
        score = (
            0.0
            if not values["case_count"]
            else float(values["dice_sum"]) / int(values["case_count"])
        )
        print(f"  {name}: {score:.4f} ({values['case_count']} cases)")


def train(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    target = args.target
    num_classes = 2 if target == "tumour" else 3
    ## new - save checkpoints every epoch
    model = UNet3D(out_channels=num_classes, base_channels=args.base_channels).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Initialize training state
    best_dice = -1.0
    start_epoch = 1

    args.model.parent.mkdir(parents=True, exist_ok=True)

    # Resume from previous training checkpoint if requested
    if args.resume:
        print(f"Loading training checkpoint: {args.checkpoint}")

        checkpoint = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]

        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

        if (
            torch.cuda.is_available()
            and checkpoint["cuda_rng_state"] is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        print(f"Resuming from epoch {start_epoch}")

    # Begin training
    for epoch in range(start_epoch, args.epochs + 1):
    ##new ends
        model.train()
        case_names = list(splits["train"])
        rng.shuffle(case_names)
        patch_batch: list[np.ndarray] = []
        label_batch: list[np.ndarray] = []
        losses: list[float] = []

        def update_model() -> None:
            inputs = torch.from_numpy(np.stack(patch_batch)[:, None]).to(device=device)
            labels = torch.from_numpy(np.stack(label_batch).astype(np.int64)).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = args.ce_weight * F.cross_entropy(logits, labels)
            loss = loss + args.dice_weight * _dice_loss(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            patch_batch.clear()
            label_batch.clear()

        for case_index, case_name in enumerate(case_names, start=1):
            if args.verbose:
                print(
                    f"  Processing training case "
                    f"{case_index}/{len(case_names)}: {case_name}",
                    flush=True,
                )

            image, label = _load_case(args.data, case_name, target)

            if args.verbose:
                print(
                    f"    Image shape: {image.shape}",
                    flush=True,
                )

            foreground_coordinates = {
                int(class_id): np.argwhere(label == class_id)
                for class_id in np.unique(label)
                if class_id != 0
            }
            for _ in range(args.patches_per_case):
                patch, label_patch = _sample_patch(
                    image,
                    label,
                    tuple(args.patch_size),
                    args.foreground_probability,
                    rng,
                    foreground_coordinates,
                )
                patch_batch.append(patch)
                label_batch.append(label_patch)
                if len(patch_batch) == args.batch_size:
                    update_model()
        if patch_batch:
            update_model()

        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch}/{args.epochs} - loss: {mean_loss:.4f}")
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            validation_dice, totals = _validate(
                model, splits["val"], args, target, device
            )
            print(f"Validation mean foreground Dice: {validation_dice:.4f}")
            _print_dice(totals, target)
            if validation_dice > best_dice:
                best_dice = validation_dice
                config = {
                    "target": target,
                    "patch_size": tuple(args.patch_size),
                    "base_channels": args.base_channels,
                    "epochs": args.epochs,
                    "patches_per_case": args.patches_per_case,
                    "batch_size": args.batch_size,
                    "foreground_probability": args.foreground_probability,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dice_weight": args.dice_weight,
                    "ce_weight": args.ce_weight,
                    "seed": args.seed,
                    "validation_overlap": args.overlap,
                    "best_validation_dice": best_dice,
                }
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "train_cases": tuple(splits["train"]),
                    },
                    args.model,
                )
                print(f"Saved best checkpoint to {args.model}")
        ##new - save checkpoints every epoch
        save_training_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            epoch,
            best_dice,
            rng,
        )
        ##new ends

def predict(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    device = _resolve_device(args.device)
    checkpoint: dict[str, Any] = torch.load(
        args.model, map_location=device, weights_only=True
    )
    config = checkpoint["config"]
    target = config["target"]
    num_classes = 2 if target == "tumour" else 3
    model = UNet3D(
        out_channels=num_classes, base_channels=int(config["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    saved_patch_size = config["patch_size"]
    if not isinstance(saved_patch_size, (list, tuple)) or len(saved_patch_size) != 3:
        raise ValueError("Model checkpoint must contain a three-dimensional patch size")
    patch_size = (
        int(saved_patch_size[0]),
        int(saved_patch_size[1]),
        int(saved_patch_size[2]),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    class_ids = (1,) if target == "tumour" else (1, 2)
    totals = {class_id: {"dice_sum": 0.0, "case_count": 0} for class_id in class_ids}

    case_names = splits[args.split]

    for case_index, case_name in enumerate(case_names, start=1):
        if args.verbose:
            print(
                f"Predicting case {case_index}/{len(case_names)}: {case_name}",
                flush=True,
            )

        image, truth = _load_case(args.data, case_name, target)

        if args.verbose:
            print(f"  Image shape: {image.shape}", flush=True)
        
        prediction = _predict_volume(
            model,
            image,
            patch_size,
            args.overlap,
            args.batch_size,
            device,
        )
        if args.verbose:
            print(f"  Prediction complete: {prediction.shape}", flush=True)

        for class_id, score in _score_case(truth, prediction, target).items():
            if score is not None:
                totals[class_id]["dice_sum"] += score
                totals[class_id]["case_count"] += 1
        if target == "tumour":
            prediction *= 2
        prediction_path = args.out / f"{case_name}.npz"
        np.savez_compressed(prediction_path, prediction=prediction)
        print(f"Predicted {case_name} -> {prediction_path}")

    print(f"Mean per-case Dice on the {args.split} split:")
    _print_dice(totals, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and save the 3D U-Net")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument(
        "--target",
        choices=("multiclass", "tumour"),
        default="multiclass",
        help="segment background/pancreas/tumour or tumour versus everything else",
    )
    train_parser.add_argument("--patch-size", type=int, nargs=3, default=DEFAULT_PATCH_SIZE)
    train_parser.add_argument("--base-channels", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--patches-per-case", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--foreground-probability", type=float, default=0.75)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--dice-weight", type=float, default=1.0)
    train_parser.add_argument("--ce-weight", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    train_parser.add_argument("--validation-every", type=int, default=5)
    train_parser.add_argument("--overlap", type=float, default=0.5)
    train_parser.add_argument("--inference-batch-size", type=int, default=1)
    # checkpoint and resume arguments for training
    train_parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("data/models/unet3d_training.pt"),
    help="Path to the training-state checkpoint",
)
    train_parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from a saved checkpoint",
)
    # verbose argument for training
    train_parser.add_argument(
    "--verbose",
    action="store_true",
    help="print detailed training and validation progress",
    )
    train_parser.set_defaults(func=train)

    predict_parser = commands.add_parser(
        "predict", help="predict a labelled split and report per-case Dice"
    )
    predict_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    predict_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    predict_parser.add_argument(
        "--out", type=Path, default=Path("data/predictions/unet3d")
    )
    predict_parser.add_argument("--batch-size", type=int, default=1)
    predict_parser.add_argument("--overlap", type=float, default=0.5)
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    predict_parser.add_argument(
    "--verbose",
    action="store_true",
    help="print detailed prediction progress",
    )
    predict_parser.set_defaults(func=predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        if args.epochs <= 0 or args.patches_per_case <= 0 or args.batch_size <= 0:
            raise ValueError("Epoch, patch, and batch counts must be positive")
        if args.base_channels <= 0 or args.validation_every <= 0:
            raise ValueError("Base channels and validation frequency must be positive")
        if any(size <= 0 for size in args.patch_size):
            raise ValueError("Patch dimensions must be positive")
        if not 0 <= args.foreground_probability <= 1:
            raise ValueError("--foreground-probability must be between 0 and 1")
        if args.learning_rate <= 0 or args.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative")
        if args.ce_weight < 0 or args.dice_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if args.ce_weight + args.dice_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        if args.inference_batch_size <= 0:
            raise ValueError("--inference-batch-size must be positive")
    elif args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    args.func(args)


if __name__ == "__main__":
    main()
