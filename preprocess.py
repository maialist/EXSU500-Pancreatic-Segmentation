"""Preprocess the Medical Segmentation Decathlon pancreas CT data (.nii.gz).

Requires: pip install nibabel numpy scipy

Pipeline per case:
  1. Load image + label, reorient to RAS+
  2. Resample to a common voxel spacing (linear for CT, nearest for labels)
  3. Clip HU to a soft-tissue window and scale to [0, 1]
  4. Save as compressed .npz (float16 image, uint8 label) in the output folder

Also writes a patient-level train/val/test split (splits.json) so that all three
models (SVM, random forest, 3D UNet) are evaluated on identical cases.

Usage:
  python preprocess.py --raw "/path/to/EXSU500_PancreasSeg_raw_data" --out data/processed
"""
import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

DEFAULT_RAW = Path.home() / "Downloads/McGill/Master's/EXSU 500/EXSU500_PancreasSeg_raw_data"
HU_MIN, HU_MAX = -100.0, 240.0  # soft-tissue window
TARGET_SPACING = (1.5, 1.5, 2.5)  # mm, RAS axes


def load_ras(path):
    """Load a NIfTI file reoriented to RAS+; returns (array, spacing)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.header.get_zooms()[:3]


def resample(vol, spacing, target, order):
    factors = [s / t for s, t in zip(spacing, target)]
    return zoom(vol, factors, order=order)


def preprocess_case(image_path, label_path):
    img, spacing = load_ras(image_path)
    lbl, _ = load_ras(label_path)
    img = resample(img, spacing, TARGET_SPACING, order=1)
    lbl = resample(lbl, spacing, TARGET_SPACING, order=0).astype(np.uint8)
    img = (np.clip(img, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return img.astype(np.float16), lbl


def make_splits(names, val_frac, test_frac, seed):
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n_test = round(len(names) * test_frac)
    n_val = round(len(names) * val_frac)
    return {
        "test": names[:n_test],
        "val": names[n_test:n_test + n_val],
        "train": names[n_test + n_val:],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="process only N cases (debug)")
    args = p.parse_args()

    images = sorted(p for p in (args.raw / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))
    if args.limit:
        images = images[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    done = []
    for i, img_path in enumerate(images, 1):
        name = img_path.name[: -len(".nii.gz")]
        label_path = args.raw / "labelsTr" / img_path.name
        out_path = args.out / f"{name}.npz"
        if not label_path.exists():
            print(f"[skip] no label for {name}")
            continue
        if not out_path.exists():
            img, lbl = preprocess_case(img_path, label_path)
            np.savez_compressed(out_path, image=img, label=lbl)
        done.append(name)
        print(f"[{i}/{len(images)}] {name}")

    splits = make_splits(done, args.val_frac, args.test_frac, args.seed)
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items()})


def load_case(path):
    """Helper for training scripts: returns (image float32, label uint8)."""
    d = np.load(path)
    return d["image"].astype(np.float32), d["label"]


if __name__ == "__main__":
    main()
