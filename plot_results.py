"""Plot Dice results produced by evaluate.py.

Figures (saved to --out-dir):
  dice_by_split.png      grouped bars of mean Dice (+/- std) per model and split, one panel per structure
  dice_distribution.png  per-case Dice box plots with points, per model, one panel per split x structure
  dice_vs_tumour_size.png  tumour Dice vs ground-truth tumour volume per case (test/val cases)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_ORDER = ["train", "val", "test"]
STRUCTURE_ORDER = ["pancreas", "tumour", "pancreas_total"]
MODEL_ORDER = ["svm", "random_forest", "unet3d"]


def _order(values, preferred):
    return [v for v in preferred if v in set(values)]


def plot_by_split(df: pd.DataFrame, out: Path) -> None:
    structures = _order(df["structure"], STRUCTURE_ORDER)
    models = _order(df["model"], MODEL_ORDER)
    splits = _order(df["split"], SPLIT_ORDER)
    fig, axes = plt.subplots(1, len(structures), figsize=(5 * len(structures), 4), sharey=True, squeeze=False)
    width = 0.8 / len(models)
    for ax, structure in zip(axes[0], structures):
        for i, model in enumerate(models):
            sub = df[(df["structure"] == structure) & (df["model"] == model)]
            grouped = sub.groupby("split")["dice"]
            means = grouped.mean().reindex(splits)
            stds = grouped.std().reindex(splits)
            x = np.arange(len(splits)) + (i - (len(models) - 1) / 2) * width
            ax.bar(x, means.fillna(0), width, yerr=stds.fillna(0), capsize=3, label=model)
        ax.set_xticks(np.arange(len(splits)))
        ax.set_xticklabels(splits)
        ax.set_title(structure)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean per-case Dice")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(out / "dice_by_split.png", dpi=200)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, out: Path) -> None:
    structures = _order(df["structure"], STRUCTURE_ORDER)
    models = _order(df["model"], MODEL_ORDER)
    splits = _order(df["split"], SPLIT_ORDER)
    fig, axes = plt.subplots(
        len(structures), len(splits), figsize=(4.5 * len(splits), 3.5 * len(structures)),
        sharey=True, squeeze=False,
    )
    rng = np.random.default_rng(0)
    for r, structure in enumerate(structures):
        for c, split in enumerate(splits):
            ax = axes[r][c]
            data = [
                df[(df["structure"] == structure) & (df["split"] == split) & (df["model"] == m)]["dice"].dropna().values
                for m in models
            ]
            ax.boxplot(data, showfliers=False)
            for k, values in enumerate(data, start=1):
                ax.scatter(k + rng.uniform(-0.15, 0.15, len(values)), values, s=8, alpha=0.5)
            ax.set_xticks(range(1, len(models) + 1))
            ax.set_xticklabels(models, rotation=20)
            ax.set_ylim(0, 1)
            ax.set_title(f"{structure} - {split}")
            if c == 0:
                ax.set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(out / "dice_distribution.png", dpi=200)
    plt.close(fig)


def plot_vs_tumour_size(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df["structure"] == "tumour") & (df["truth_voxels"] > 0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in _order(sub["model"], MODEL_ORDER):
        m = sub[sub["model"] == model]
        ax.scatter(m["truth_voxels"], m["dice"], s=14, alpha=0.6, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("Ground-truth tumour volume (voxels)")
    ax.set_ylabel("Tumour Dice")
    ax.set_ylim(-0.02, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "dice_vs_tumour_size.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/results/dice_per_case.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_by_split(df, args.out_dir)
    plot_distribution(df, args.out_dir)
    plot_vs_tumour_size(df, args.out_dir)
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
