"""Train and run a voxel-wise SVM baseline for pancreas CT segmentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import LinearSVC

CLASS_IDS = (0, 1, 2)
CLASS_NAMES = {0: "background", 1: "pancreas", 2: "tumour"}
NEIGHBOUR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_PATH = Path("data/models/svm.joblib")


def target_labels(label: np.ndarray, target: str) -> np.ndarray:
    """Return labels for multiclass segmentation or tumour-vs-rest."""
    if target == "multiclass":
        return label
    if target == "tumour":
        return (label == 2).astype(np.uint8)
    raise ValueError(f"Unsupported target: {target}")


def extract_features(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Extract the edge-padded 3x3x3 intensity patch around each voxel."""
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_voxels, 3)")

    coordinates = coordinates.astype(np.intp, copy=False)
    if coordinates.size and (
        np.any(coordinates < 0)
        or any(np.any(coordinates[:, axis] >= image.shape[axis]) for axis in range(3))
    ):
        raise ValueError("Voxel coordinates are outside the image bounds")

    features = np.empty((len(coordinates), len(NEIGHBOUR_OFFSETS)), dtype=np.float32)
    for column, offset in enumerate(NEIGHBOUR_OFFSETS):
        indices = tuple(
            np.clip(coordinates[:, axis] + offset[axis], 0, image.shape[axis] - 1)
            for axis in range(3)
        )
        features[:, column] = image[indices]
    return features


def _sample_training_data(
    data_dir: Path,
    case_names: list[str],
    target: str,
    samples_per_class: int,
    samples_per_class_per_case: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_ids = (0, 1) if target == "tumour" else CLASS_IDS
    class_features: dict[int, list[np.ndarray]] = {class_id: [] for class_id in class_ids}

    for case_name in case_names:
        case_path = data_dir / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            labels = target_labels(case["label"], target)
        if image.shape != labels.shape:
            raise ValueError(
                f"Image and label shapes differ for {case_name}: "
                f"{image.shape} != {labels.shape}"
            )

        flat_labels = labels.ravel()
        for class_id in class_ids:
            voxels = np.flatnonzero(flat_labels == class_id)
            count = min(len(voxels), samples_per_class_per_case)
            if count == 0:
                continue
            selected = rng.choice(voxels, size=count, replace=False)
            coordinates = np.column_stack(np.unravel_index(selected, labels.shape))
            class_features[class_id].append(extract_features(image, coordinates))

    features_by_class: list[np.ndarray] = []
    labels_by_class: list[np.ndarray] = []
    for class_id in class_ids:
        samples = class_features[class_id]
        if not samples:
            raise ValueError(
                f"No training voxels for class {class_id}; check the labels and split"
            )
        class_data = np.concatenate(samples, axis=0)
        count = min(len(class_data), samples_per_class)
        if count < len(class_data):
            selected = rng.choice(len(class_data), size=count, replace=False)
            class_data = class_data[selected]
        features_by_class.append(class_data)
        labels_by_class.append(np.full(count, class_id, dtype=np.uint8))

    return np.concatenate(features_by_class), np.concatenate(labels_by_class)


def _load_splits(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as split_file:
        splits = json.load(split_file)
    required = ("train", "val", "test")
    if not all(isinstance(splits.get(name), list) for name in required):
        raise ValueError(f"{path} must contain train, val, and test case lists")
    if any(not names for names in (splits[name] for name in required)):
        raise ValueError("Train, validation, and test splits must all be non-empty")
    if any(not isinstance(case, str) for name in required for case in splits[name]):
        raise ValueError("Split entries must be case-name strings")
    if len(set(splits["train"] + splits["val"] + splits["test"])) != sum(
        len(splits[name]) for name in required
    ):
        raise ValueError("Patient cases overlap between train, validation, and test")
    return splits


def _predict_volume(
    classifier: LinearSVC, image: np.ndarray, batch_size: int
) -> np.ndarray:
    prediction = np.empty(image.shape, dtype=np.uint8)
    flat_prediction = prediction.ravel()
    voxel_count = image.size
    for start in range(0, voxel_count, batch_size):
        flat_indices = np.arange(start, min(start + batch_size, voxel_count))
        coordinates = np.column_stack(np.unravel_index(flat_indices, image.shape))
        features = extract_features(image, coordinates)
        flat_prediction[start : start + len(flat_indices)] = classifier.predict(features)
    return prediction


def train(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    features, labels = _sample_training_data(
        args.data,
        splits["train"],
        args.target,
        args.samples_per_class,
        args.samples_per_class_per_case,
        args.seed,
    )
    classifier = LinearSVC(
        C=args.c,
        class_weight="balanced",
        dual=False,
        loss="squared_hinge",
        max_iter=5000,
        random_state=args.seed,
    )
    classifier.fit(features, labels)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "target": args.target,
            "class_ids": (0, 1) if args.target == "tumour" else CLASS_IDS,
            "config": {
                "c": args.c,
                "samples_per_class": args.samples_per_class,
                "samples_per_class_per_case": args.samples_per_class_per_case,
                "seed": args.seed,
                "feature": "edge-padded 3x3x3 normalized CT intensity patch",
            },
            "train_cases": tuple(splits["train"]),
        },
        args.model,
    )
    print(f"Trained on {len(labels):,} sampled voxels; saved model to {args.model}")
    for class_id in np.unique(labels):
        class_name = (
            ("non-tumour" if class_id == 0 else "tumour")
            if args.target == "tumour"
            else CLASS_NAMES[int(class_id)]
        )
        print(f"  {class_name}: {(labels == class_id).sum():,}")


def predict(args: argparse.Namespace) -> None:
    splits = _load_splits(args.splits)
    artifact = joblib.load(args.model)
    classifier: LinearSVC = artifact["classifier"]
    target = artifact["target"]
    class_ids = tuple(artifact["class_ids"])
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {
        class_id: {"dice_sum": 0.0, "case_count": 0}
        for class_id in class_ids
        if class_id != 0
    }

    for case_name in splits[args.split]:
        case_path = args.data / f"{case_name}.npz"
        if not case_path.is_file():
            raise FileNotFoundError(f"Processed case is missing: {case_path}")
        with np.load(case_path) as case:
            image = case["image"].astype(np.float32, copy=False)
            truth = target_labels(case["label"], target)
        if image.shape != truth.shape:
            raise ValueError(
                f"Image and label shapes differ for {case_name}: "
                f"{image.shape} != {truth.shape}"
            )
        prediction = _predict_volume(classifier, image, args.batch_size)

        for class_id, values in totals.items():
            truth_mask = truth == class_id
            prediction_mask = prediction == class_id
            denominator = int(truth_mask.sum() + prediction_mask.sum())
            if denominator:
                values["dice_sum"] += (
                    2.0 * int(np.logical_and(truth_mask, prediction_mask).sum()) / denominator
                )
                values["case_count"] += 1

        if target == "tumour":
            prediction *= 2
        prediction_path = args.out / f"{case_name}.npz"
        np.savez_compressed(prediction_path, prediction=prediction)
        print(f"Predicted {case_name} -> {prediction_path}")

    print(f"Mean per-case Dice on the {args.split} split:")
    for class_id, values in totals.items():
        class_name = "tumour" if target == "tumour" else CLASS_NAMES[class_id]
        score = (
            0.0
            if values["case_count"] == 0
            else values["dice_sum"] / values["case_count"]
        )
        print(f"  {class_name}: {score:.4f} ({values['case_count']} cases)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="fit and save the SVM model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument(
        "--target",
        choices=("multiclass", "tumour"),
        default="multiclass",
        help="segment background/pancreas/tumour or tumour versus everything else",
    )
    train_parser.add_argument("--c", type=float, default=1.0)
    train_parser.add_argument("--samples-per-class", type=int, default=10_000)
    train_parser.add_argument("--samples-per-class-per-case", type=int, default=250)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    predict_parser = commands.add_parser(
        "predict", help="predict a labelled split and report mean per-case Dice"
    )
    predict_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    predict_parser.add_argument("--splits", type=Path, default=DEFAULT_DATA_DIR / "splits.json")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--split", choices=("val", "test"), required=True)
    predict_parser.add_argument("--out", type=Path, default=Path("data/predictions/svm"))
    predict_parser.add_argument("--batch-size", type=int, default=100_000)
    predict_parser.set_defaults(func=predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        if args.c <= 0:
            raise ValueError("--c must be positive")
        if args.samples_per_class <= 0 or args.samples_per_class_per_case <= 0:
            raise ValueError("Sample limits must be positive")
    elif args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
