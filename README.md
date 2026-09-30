# EchoLVEF v7-turbo — EchoRisk-MICCAI 2026 Task 1 (LVEF Regression)

---

## A. Algorithm

LVEF (Left Ventricular Ejection Fraction) is estimated from echocardiographic cine videos in two views: A4C and A2C. The proposed framework consists of a three-stage backbone with multi-signal fusion.

## A.1 Backbone (Appearance Representation)

### **Stage 1 — EchoNet ED/ES Contrastive Pretraining** (`train_backbone.run_stage1`)

Built on top of EchoPrime MViT-v2-s, contrastive learning is applied using:

`ReLU(cos(f_ED, f_ES) − 0.70)`

to push end-diastolic (ED) and end-systolic (ES) features apart, preventing temporal feature collapse. EchoNet data introduces no leakage with respect to MICCAI labels.

Output:

`weights/backbone_s1.pt`

### **Stage 2 — MICCAI MTL LoRA Fine-tuning** (`run_stage2_honest`)

- Frozen backbone + LoRA (r=8, blocks 12–15)
- Dual-view temporal attention
- Dual-head prediction for EF and biomarkers

Epoch selection is performed using a patient-grouped INNER-VAL split from the MICCAI training set. The validation set is used only once for final scoring.

Output:

`weights/backbone.pt`

(the deployed appearance encoder)

During inference:

- Each video: 5 clips × 16 frames
- Feature dimension: 512-d
- Statistical pooling:

`mean | max | std`

producing:

- 1536-d representation per view

---

## A.2 Three Signal Branches (`extract.py`)

The geometry and motion branches are label-free with respect to MICCAI labels.

### **Appearance**

The statpool features from the backbone above.

### **Geometry**

Two ResNet34-UNet LV segmentation models:

- Dice score: 0.937
- Dice score: 0.938

Processing:

- Determine ED/ES frames according to LV cavity area
- Compute EF using Simpson’s biplane method
- Prefer biplane estimation; otherwise use single-view estimation

### **Motion**

EchoNet R(2+1)D-18 network:

- Direct EF prediction from motion representation

### **CAMUS Augmentation**

Appearance statpool features are extracted from CAMUS semi-sequences to provide broader distribution augmentation.

---

## A.3 Fusion (`fusion.py` / `heads_bundle.py`)

All parameters are selected using TRAIN OOF (out-of-fold) validation only.

### **Appearance Routing Ridge Heads**

Separate ridge regression heads:

- Biplane
- A4C
- A2C
- Low-EF expert
- High-EF expert

Features include:

- CAMUS augmentation
- Patient-grouped 5-fold OOF training
- Closed-form ridge regression

The ridge coefficient α is selected using train OOF performance.

### **Consensus Blend**

Geometry and motion predictions are:

1. z-score normalized
2. Converted to EF scale

Consensus prediction:

`con`

Final prediction:

`(1−w) · app + w · con`

where `w` is independently selected for:

- biplane cases
- single-view cases

### **Expert Trust Mechanism**

For low/high EF ranges in single-view cases:

- Route predictions through low/high EF expert heads according to thresholds.

### **Sibling Anchor PSA**

For single-view exams:

- Use the predicted EF from the same patient’s biplane sibling exam
- Used as a temporal anchor
- No labels involved

### **Inference Bundle**

All parameters:

- Regression heads
- z-score transformations
- Blend weights
- Expert routing thresholds
- PSA parameters

are fitted offline into:

`weights/infer_bundle.npz`

Inference only applies the precomputed parameters without retraining.

---

# B. Performance and Runtime

## B.1 Accuracy (MICCAI Validation Set, n=156)

Official `score.py`, without TTA:

| Metric | Value |
|---|---:|
| **MAE (primary)** | **4.0342** |
| RMSE | 5.146 |
| Pearson r | 0.699 |
| Biplane (n=144) / Single-view (n=12) MAE | 3.92 / 4.41 |
| Slope / Bias | 0.37 / +0.19 |

---

## B.2 Runtime

Hardware:

- RTX 5090
- Validation set: n=156
- Cold-start inference

Test extrapolation:

- Approximately 340 exams
- 30% split

| Method | Validation Wall Time | Extrapolated Test Time | MAE | Difference vs Reference max\|Δ\| |
|---|---:|---:|---:|---:|
| Reference (serial decode → extract) | 186s | 6.8 min | 4.0342 | 0 (deterministic) |
| **`infer --mode exact` (default)** | 179s | **6.5 min** | 4.0343 | 0.003 EF |
| **`infer --mode batched`** | 123s | **4.5 min** | 4.0337 | 0.082 EF (1/156) |

---

# C. Usage

```bash
# Inference (cold start, no cache)
# Reproduce accuracy + timing + test extrapolation

python infer.py --split val                         # exact (default)
python infer.py --split val --mode batched          # 1.5× faster

python infer.py --split test \
    --labels <test.csv> \
    --dicom-root <dir> \
    --out sub.csv


# Evaluate a submission

python evaluate.py --csv submissions/submission_val.csv


# Training (starting from pretrained EchoPrime base)

python train.py --steps all
# s1 -> s2 -> extract -> fusion

python train.py --steps fusion
# Refit fusion heads only using cached features (fast)


# Offline fitting of cold-start inference bundle

python build_bundle.py


# Smoke test
# Verify all training-side stages run successfully
# Uses mini dataset and writes only to temporary directories

python tools/smoke_test.py


# Profile extraction sub-steps

python tools/profile_extract.py --n 16
```

---

# D. Directory Structure

```
config.py              Paths + hyperparameters (relative to ROOT)

echolvef/
    models.py          MViT backbone (LoRA/temporal attention)
                       + ResNet-UNet segmentation models

    data.py            Clip sampling +
                       EchoNet/MICCAI/CAMUS datasets

    train_backbone.py  Stage 1 contrastive training /
                       Stage 2 honest MTL training

    extract.py         Appearance + geometry (Simpson) +
                       motion + CAMUS feature extraction

    fusion.py          Routing ridge regression +
                       consensus blending +
                       expert routing +
                       sibling anchor +
                       official scoring

    heads_bundle.py    Complete inference bundle fitting
                       and deployment

    data_dicom.py      DICOM → (T,224,224,3) preprocessing
                       (sector mask/cropping/resize)

    turbo_core.py      GPU pipeline +
                       batched cold-start extraction core
                       (exact/batched modes)

train.py               Training orchestration
                       (s1/s2/extract/fusion)

infer.py               Cold-start inference entry point (turbo)

evaluate.py            Official scoring + per-bin analysis

build_bundle.py        Offline fitting of infer_bundle.npz

score.py               Official Task 1 scoring script
                       (self-contained copy)
```