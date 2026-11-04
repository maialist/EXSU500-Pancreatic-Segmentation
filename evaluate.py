"""Compute per-case Dice for every model on the train/val/test splits.

Reads saved predictions (data/predictions/<model>/<case>.npz, key "prediction")
and compares them with the processed labels. Cases without a saved prediction
are skipped, so run each model's `predict --split train|val|test` first.
Writes a long-format CSV consumed by plot_results.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ("svm", "random_forest", "unet3d")
SPLITS = ("train", "val", "test")
# Pancreas = label 1 only; tumour = label 2; "pancreas_total" = labels 1 and 2 merged.
STRUCTURES = {
    "pancreas": lambda x: x == 1,
    "tumour": lambda x: x == 2,
    "pancreas_total": lambda x: x > 0,
}


def dice(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Dice of two boolean masks; NaN when both are empty (undefined)."""
    denominator = int(truth.sum() + prediction.sum())
    if denominator == 0:
        return float("nan")
    return 2.0 * int(np.logical_and(truth, prediction).sum()) / denominator


def evaluate(args: argparse.Namespace) -> list[dict]:
    with args.splits.open() as handle:
        splits = json.load(handle)
    rows: list[dict] = []
    for model in args.models:
        for split in SPLITS:
            found = 0
            for case in splits[split]:
                prediction_path = args.predictions / model / f"{case}.npz"
                if not prediction_path.is_file():
                    continue
                with np.load(args.data / f"{case}.npz") as data:
                    truth = data["label"]
                with np.load(prediction_path) as data:
                    prediction = data["prediction"]
                if truth.shape != prediction.shape:
                    raise ValueError(f"Shape mismatch for {model}/{case}")
                found += 1
                for name, select in STRUCTURES.items():
                    rows.append(
                        {
                            "model": model,
                            "split": split,
                            "case": case,
                            "structure": name,
                            "dice": dice(select(truth), select(prediction)),
                            "truth_voxels": int(select(truth).sum()),
                        }
                    )
            print(f"{model:14s} {split:5s}: {found}/{len(splits[split])} cases")
    return rows


def summarise(rows: list[dict]) -> None:
    print("\nMean (std) per-case Dice; NaN cases (no truth and no prediction) ignored")
    keys = sorted({(r["model"], r["split"], r["structure"]) for r in rows})
    for model, split, structure in keys:
        values = np.array(
            [
                r["dice"]
                for r in rows
                if (r["model"], r["split"], r["structure"]) == (model, split, structure)
            ]
        )
        values = values[~np.isnan(values)]
        if values.size:
            print(
                f"{model:14s} {split:5s} {structure:15s} "
                f"{values.mean():.3f} ({values.std():.3f})  n={values.size}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--out", type=Path, default=Path("data/results/dice_per_case.csv"))
    args = parser.parse_args()

    rows = evaluate(args)
    if not rows:
        raise SystemExit("No predictions found; run each model's predict command first.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarise(rows)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
