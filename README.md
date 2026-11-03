# EXSU500-Pancreatic-Segmentation

## Project Overview

This project investigates the use of artificial intelligence to accurately identify and delineate pancreatic tumours on CT imaging. Automated tumour segmentation could provide a foundation for computer-assisted visualization of tumor anatomy during preoperative planning and future image-guided surgical systems. 

The project will involve training and comparing multiple machine learning models to determine how accurately they can perform the segmentation task.

The project is being completed as part of **EXSU 500: Fundamentals of Artificial Intelligence in Medicine** at McGill University.

The intended use of the model is **research**. It is not intended for clinical decision-making.

---

## Task Definition

**Task type:** Multi-class image segmentation

**Specific task:** Voxel-wise segmentation of background, pancreas, and pancreatic tumour

### Task Description

The models take **CT** medical images as input and predict background (label 0), pancreas (label 1), and pancreatic tumour (label 2) at the voxel level. The SVM also supports a tumour-versus-rest binary target.

---

## Dataset

Dataset from Task07 of the Medical Segmentation Decathlon - 421 3D volumes of portal venous phase CT scans of patients with pancreatic tumours. 

All scans are sourced from the Memorial Sloan Kettering Cancer Center (New York City, NY, USA). 

### Source

**Dataset provider:** Medical Segmentation Decathlon

**URL:** http://medicaldecathlon.com/

**Citation:** https://arxiv.org/abs/1902.09063 

---

## Support vector machine baseline

The voxel-wise SVM uses the shared patient-level `train`/`val`/`test` split and the preprocessed arrays. It learns from a 3x3x3 neighbourhood of normalized CT intensities and predicts each volume in batches. The default is a linear SVM: it is considerably more practical for millions of voxels than a kernel SVM while still providing a classical SVM baseline.

Install scikit-learn in the same environment as the preprocessing dependencies, then run:

```bash
python svm.py train
python svm.py predict --split val
```

Training saves `data/models/svm.joblib`. Predictions are written and mean per-case Dice is reported for the selected labelled split (cases with neither reference nor predicted voxels for a class are excluded from that class's mean). Use validation predictions for model selection; after choices are fixed, evaluate the test split once:

```bash
python svm.py predict --split test
```

Train a binary tumour-versus-rest model with `python svm.py train --target tumour`. The binary prediction is saved using the original label values (0 background/non-tumour, 2 tumour).

### Fixed comparison protocol and adjustable SVM choices

The following are fixed for a fair comparison with the other project models: source data and label definitions; patient-level split from `data/processed/splits.json`; preprocessing (RAS orientation, 1.5 x 1.5 x 2.5 mm spacing, -100 to 240 HU window, and [0, 1] intensity scaling); and using validation data for tuning and test data only for final evaluation. Do not change preprocessing or the split for the SVM alone.

The implementation fixes the linear kernel, 3x3x3 edge-padded intensity features, balanced class weights, squared-hinge loss, and a 5,000-iteration limit. Each training run can change `--target`, `--c`, `--samples-per-class`, `--samples-per-class-per-case`, and `--seed`. `--data`, `--splits`, and `--model` choose the input and output paths; prediction additionally requires `--split` (`val` or `test`) and allows `--batch-size` and `--out` to be changed. All training choices are stored in the model artifact.

Recommended starting point for this project: keep the shared preprocessing and patient split unchanged; use the default three-class target to match the project goal; start with `C=1`, at most 250 samples per class per training patient and 10,000 per class overall, seed 42. If the project question is specifically tumour detection, choose `--target tumour` and report tumour Dice. If tuning `C` or sampling limits, compare candidates on validation data only and record the selected settings before running the test set. These are reproducible baseline recommendations, not universal or clinically validated SVM standards. The dataset labels are reference annotations for this research benchmark, not an independent clinical gold standard. This project remains research-only and is not for clinical decision-making.
