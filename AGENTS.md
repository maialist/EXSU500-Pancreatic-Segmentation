# AGENTS.md

## Project goal

Compare three models of increasing expected efficacy on the same task: segmenting the pancreas and pancreatic tumours in CT scans.

1. Support vector machine (SVM)
2. Random forest
3. 3D U-Net

All three use the same data, preprocessing and patient-level split, so results are directly comparable. The wider motivation (computer-assisted visualization for preoperative planning and image-guided surgery) is in [README.md](README.md).

## Dataset

Medical Segmentation Decathlon Task07 Pancreas (portal-venous CT, Memorial Sloan Kettering), stored as `.nii.gz`.

- Raw location (not in the repo): `~/Downloads/McGill/Master's/EXSU 500/EXSU500_PancreasSeg_raw_data`
- Layout: `dataset.json`, `imagesTr/` (281 labelled cases), `imagesTs/` (139 unlabelled cases), `labelsTr/`
- Labels: 0 = background, 1 = pancreas, 2 = cancer

## What has been done so far

- [preprocess.py](preprocess.py) converts the raw NIfTI files into training-ready arrays:
  1. Load image and label with `nibabel` and reorient to RAS+ (consistent axis layout across scans).
  2. Resample to 1.5 x 1.5 x 2.5 mm voxels (common physical scale). The image uses linear interpolation and the label uses nearest-neighbour so labels stay integers 0/1/2.
  3. Clip CT values to -100..240 HU (soft tissue) and scale to [0, 1].
  4. Save each case as `data/processed/<case>.npz` (`image` float16, `label` uint8).
  5. Write `data/processed/splits.json`, a seeded (42) 70/15/15 train/val/test split by patient, drawn from `imagesTr` because `imagesTs` has no labels.
  6. `models/svm.py` implements an SVM, `models/random_forest.py` a random forest, and `models/unet3d.py` a 3D U-Net.
- [.gitignore](.gitignore) excludes raw data, `data/`, and `.nii`, `.nii.gz`, `.npz`, `.npy` files.

Run it with `python preprocess.py` (use `--raw` to override the data path, `--limit N` for a quick test). Dependencies: `nibabel`, `numpy`, `scipy`. `load_case(path)` returns `(image float32, label uint8)` for one case.

## Conventions for contributors and agents

- Never commit raw or processed data. Keep the `.gitignore` entries.
- Every model must use `splits.json`; do not create separate splits.
- Do not change preprocessing parameters (spacing, HU window) without reprocessing and re-running all models.
- Interpolate labels with nearest-neighbour only.

All three models use the same patient-level split. Tune using validation data and reserve the test split for final evaluation. The models are for research, not clinical decision-making.
