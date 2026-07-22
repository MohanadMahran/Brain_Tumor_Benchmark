# 🧠 3D Brain Tumor Segmentation Benchmark: 3D U-Net vs UNETR

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-2d335c.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, evidence-based benchmarking framework comparing 3D Convolutional Neural Networks (**3D U-Net**) and Vision Transformers (**UNETR**) for 3D multi-modal brain tumor segmentation.

Models were trained on a composite, MD5-deduplicated dataset derived from **BraTS 2021** and **BraTS 2024 Adult Glioma** (1,560 training cases, 391 validation cases), and evaluated out-of-distribution on the independent **UPenn-GBM** benchmark dataset (30 cases, exclusively Glioblastoma).

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Benchmark Results](#-benchmark-results)
- [Sample Predictions](#-sample-predictions)
- [Training Summary](#-training-summary)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Project Structure](#-project-structure)
- [Running the Pipeline](#-running-the-pipeline)
- [Configuration](#-configuration)
- [Experiment Tracking](#-experiment-tracking)
- [Design Decisions](#-design-decisions)
- [Limitations & Scope](#-limitations--scope)
- [Reproducibility](#-reproducibility)

---

## 🌟 Overview

This project provides a controlled empirical comparison between 3D CNN and 3D Vision Transformer architectures for medical image segmentation.

### Evaluated Architectures & Parameter Parity:
1. **3D U-Net**: Purely convolutional encoder-decoder network. Configured with **19,992,603 parameters** (~19.99M), utilizing `GroupNorm` (groups=4) and base channel width of 60 progressing through `[60, 120, 240, 480]`.
2. **UNETR (UNet Transformers)**: Vision Transformer encoder with CNN decoder. Configured with **19,327,907 parameters** (~19.33M), utilizing `LayerNorm`, an embedding dimension of 384, 6 transformer layers, 8 attention heads, and $16 \times 16 \times 16$ patch tokenization. Incorporates PyTorch gradient checkpointing.

> **Core Fairness Constraint**: The parameter counts of both models differ by only **3.44%** ($|19.99\text{M} - 19.33\text{M}| / 19.33\text{M}$), satisfying the project's strict $\le 5\%$ parameter parity tolerance constraint for fair architectural evaluation.

Both models accept 4-channel input MRI scans (T1, T1Gd, T2, FLAIR) preprocessed to $128 \times 128 \times 128$ spatial patch resolution and predict 3 overlapping tumor sub-regions:
- **Enhancing Tumor (ET)** — Label 4
- **Tumor Core (TC)** — Labels 1 + 4
- **Whole Tumor (WT)** — Labels 1 + 2 + 4

---

## ✨ Key Features

- **Empirical Rigor**: Strict parameter matching (~19.3M–19.99M) trained under identical optimizers, schedulers, augmentations, and effective batch sizes.
- **Hardware Optimization**: FP16 Mixed Precision (AMP), Gradient Checkpointing (UNETR), and 4-step Gradient Accumulation enable efficient execution on single GPUs.
- **Data Integrity**: MD5-hash cross-year deduplication prevents data leakage between BraTS 2021 and BraTS 2024 datasets.
- **Robust Evaluation**: Standardized 3D Dice score and 95th percentile Hausdorff Distance (HD95) with label remapping ($3 \to 4$) for out-of-distribution benchmarks.
- **Full Traceability**: Automated MLflow logging for metrics, hyperparameters, VRAM allocations, and SLURM wall-time safety checkpoints.

---

## 📊 Benchmark Results

Both models were evaluated on the independent **UPenn-GBM** out-of-distribution benchmark cohort (30 cases, 4 MRI modalities each). The key quantitative metrics and generalization performance are summarized below:

### UPenn-GBM Evaluation Performance (30 Cases)

| Metric / Dimension | 3D U-Net | UNETR | Delta (U-Net vs UNETR) | Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Model Parameters** | 19,992,603 (~19.99M) | 19,327,907 (~19.33M) | +664,696 (+3.44%) | Parity Held (<5%) |
| **BraTS Val Mean Dice (391 cases)** | **0.8694** | 0.8362 | +0.0332 (+3.32%) | **3D U-Net** |
| **UPenn Mean Dice** | **0.5186 ± 0.3223** | 0.4749 ± 0.3007 | +0.0437 (+4.37%) | **3D U-Net** |
| **UPenn ET Dice (Enhancing Tumor)** | **0.5188 ± 0.3809** | 0.4454 ± 0.3223 | +0.0734 (+7.34%) | **3D U-Net** |
| **UPenn TC Dice (Tumor Core)** | 0.4855 ± 0.3658 | **0.4976 ± 0.3387** | -0.0121 (-1.21%) | **UNETR** |
| **UPenn WT Dice (Whole Tumor)** | **0.5514 ± 0.2616** | 0.4818 ± 0.2887 | +0.0696 (+6.96%) | **3D U-Net** |
| **UPenn Mean HD95 (mm)** | 90.58 ± 31.52 | **87.71 ± 27.26** | +2.87 mm | **UNETR** |
| **UPenn ET HD95 (mm)** | **73.43 ± 54.37** | 74.94 ± 43.55 | -1.51 mm | **3D U-Net** |
| **UPenn TC HD95 (mm)** | 105.50 ± 32.90 | **93.10 ± 32.62** | +12.40 mm | **UNETR** |
| **UPenn WT HD95 (mm)** | **92.81 ± 20.13** | 95.08 ± 17.67 | -2.27 mm | **3D U-Net** |
| **Generalization Gap (`dice_drop_vs_val`)** | **35.08%** (0.8694 → 0.5186) | 36.13% (0.8362 → 0.4749) | -1.05% drop | **3D U-Net** |
| **Inference Time per Case** | 4.58s ± 0.42s | **1.84s ± 0.19s** | -2.74s (-59.8%) | **UNETR (2.48x faster)** |
| **Evaluation Peak VRAM** | 3,718 MB (~3.72 GB) | **1,979 MB (~1.98 GB)** | -1,739 MB (-46.8%) | **UNETR** |

### Key Scientific Findings & Interpretation:

1. **Generalization Gap (Domain Shift)**:
   - Both models experienced severe performance degradation when evaluated out-of-distribution on UPenn-GBM compared to their BraTS validation scores (**>35 percentage points drop**).
   - **3D U-Net** demonstrated higher domain robustness, retaining a mean Dice of **0.5186** compared to UNETR's **0.4749** (+4.37% absolute improvement). Its generalization gap was slightly smaller (**35.08%** vs **36.13%**).

2. **Region-Specific Performance Dynamics**:
   - **3D U-Net** outperformed UNETR substantially on **Enhancing Tumor (ET: 0.5188 vs 0.4454)** and **Whole Tumor (WT: 0.5514 vs 0.4818)** boundary segmentation. Local inductive biases of 3D convolutions preserve fine-grained spatial contrast and high-frequency edge details better across scanner shifts.
   - **UNETR** achieved slightly superior performance on **Tumor Core (TC: 0.4976 vs 0.4855)** and produced lower boundary distance errors for the tumor core (**TC HD95: 93.10 mm vs 105.50 mm**), leveraging global self-attention to model long-range context across necrotic sub-regions.

3. **Computational Efficiency**:
   - **UNETR** proved significantly more computationally efficient during sliding-window inference, taking only **1.84 seconds per case** (**2.48x faster** than 3D U-Net's 4.58s) and consuming **46.8% less peak VRAM** (1.98 GB vs 3.72 GB).

4. **Statistical Limitation**:
   - With $N = 30$ benchmark cases, sample variance is high ($\text{std} \approx 0.30 - 0.38$). Extremely difficult clinical cases with severe tissue distortion or atypical scanner contrast (e.g., `UPENN-GBM-00307` and `UPENN-GBM-00388`) yielded near-zero Dice scores for both models.

---

## 🖼️ Sample Predictions

Below are the 3D axial slice traversal animations for case `UPENN-GBM-00307` from the UPenn-GBM benchmark, comparing Ground Truth annotations against 3D U-Net and UNETR model predictions:

| Ground Truth (GT) | 3D U-Net Prediction | UNETR Prediction |
| :---: | :---: | :---: |
| ![Ground Truth](gifs/segm_t1.gif) | ![3D U-Net](gifs/unet3d_t1.gif) | ![UNETR](gifs/unetr_t1.gif) |
| **Ground Truth Mask** | **3D U-Net Prediction** | **UNETR Prediction** |

---

## 📈 Training Summary

Training was executed on High-Performance Computing (HPC) infrastructure with SLURM wall-time safety preemption.

### Training Progress & Timeline

| Parameter / Milestone | 3D U-Net | UNETR |
| :--- | :---: | :---: |
| **Configured `num_epochs`** | 500 | 500 |
| **Epochs Actually Completed** | **500** (Epochs 0 to 499) | **500** (Epochs 0 to 499) |
| **Early Stopping Triggered** | **No** (`early_stopped: False`) | **No** (`early_stopped: False`) |
| **Patience Limit** | 30 epochs | 30 epochs |
| **SLURM Resumes Count** | 4 Resumes (Jobs 1735919 → 1745082 → 1747730 → 1750715 → 1751820) | 3 Resumes (Jobs 1737596 → 1745081 → 1747729 → 1750014) |
| **Peak Training VRAM** | **19,025 MiB** (~19.03 GB) | **7,026 MiB** (~7.03 GB) |
| **Best Val Mean Dice (BraTS)** | **0.8694** (Achieved at **Epoch 419**) | **0.8362** (Achieved at **Epoch 469**) |
| **Final Checkpoint Val Dice** | 0.8668 (Epoch 499) | 0.8361 (Epoch 499) |

### Training Loss & Learning Rate Schedule:
- **Loss Function**: Combined DiceCE Loss (Soft Dice Loss + Cross Entropy).
- **Optimization**: AdamW optimizer ($1.0\times 10^{-5}$ weight decay).
- **3D U-Net LR**: Initial base $\text{LR} = 3.0 \times 10^{-4}$, 10-epoch linear warmup, followed by Cosine Annealing decay down to $\text{min\_lr} = 1.0 \times 10^{-6}$ at epoch 500.
- **UNETR LR**: Initial base $\text{LR} = 1.0 \times 10^{-4}$, 15-epoch linear warmup, followed by Cosine Annealing decay down to $\text{min\_lr} = 1.0 \times 10^{-6}$ at epoch 500.
- Both models exhibited smooth loss decay (3D U-Net: $0.60+ \to 0.279$; UNETR: $0.65+ \to 0.279$) without numerical instability or gradient explosions.

---

## 🛠 Installation

This project uses [Astral UV](https://github.com/astral-sh/uv) for fast, deterministic dependency resolution.

1. **Install UV:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/MohanadMahran/Brain_Tumor_Benchmark.git
   cd Brain_Tumor_Benchmark
   ```

3. **Synchronize Dependencies:**
   ```bash
   uv sync
   ```

---

## 📂 Data Preparation

Data must be structured under `data/raw/`:

### 1. Training Set (BraTS 2021 + BraTS 2024)
- **BraTS 2021**: 1,251 multi-modal cases.
- **BraTS 2024 Adult Glioma**: 700 cases.
- **Deduplication**: Executed via MD5 hash comparison on raw `.nii.gz` volumes (`verify_deduplicate.py`). Yielded 1,951 unique pool cases (0 duplicates found across sets). Split deterministically into **1,560 training cases** and **391 validation cases** (80/20 split, seed 42).

### 2. Independent Out-of-Distribution Benchmark (UPenn-GBM)
- **UPenn-GBM**: 30 cases from the UPenn-GBM TCIA collection (exclusively Glioblastoma).
- **Label Remapping**: UPenn-GBM encodes Enhancing Tumor as label **3**, whereas BraTS uses label **4**. The data loader automatically applies `3 → 4` label remapping during volume preprocessing (`data/preprocessing.py`).

```text
data/raw/
├── brats2021/          # BraTS 2021 Training Data (1,000 train cases)
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
├── brats2024/          # BraTS 2024 Adult Glioma Data (560 train cases)
│   └── ...
└── UPenn-GBM/          # UPenn-GBM Benchmark (30 cases, label 3->4 remapped)
    ├── UPENN-GBM-00307/
    │   ├── UPENN-GBM-00307_11_T1.nii.gz
    │   ├── UPENN-GBM-00307_11_T1GD.nii.gz
    │   ├── UPENN-GBM-00307_11_T2.nii.gz
    │   ├── UPENN-GBM-00307_11_FLAIR.nii.gz
    │   └── UPENN-GBM-00307_11_segm.nii.gz
    └── ...
```

---

## 🚀 Running the Pipeline

The project provides a unified CLI via `main.py`.

### 1. Data Verification & Deduplication
```bash
uv run python main.py --mode prepare
```

### 2. Model Training
To launch training for 3D U-Net or UNETR:
```bash
# Train 3D U-Net
uv run python scripts/train_unet3d.py --config configs/unet3d.yaml --base_config configs/base.yaml

# Train UNETR
uv run python scripts/train_unetr.py --config configs/unetr.yaml --base_config configs/base.yaml
```

### 3. Out-of-Distribution Benchmark Evaluation
To run evaluation on the UPenn-GBM benchmark:
```bash
uv run python main.py --mode test
```

### 4. 3D Mask Prediction & Inference
To generate 3D NIfTI segmentation predictions on a single sample:
```bash
uv run python outputs/results/predict_masks.py
```

---

## ⚙️ Configuration

Decoupled configuration parameters are stored in `configs/`:

- `base.yaml`: Shared settings (seed `42`, patch size `[128, 128, 128]`, physical batch size `2`, gradient accumulation `4` -> effective batch size `8`, warmup epochs, learning rate schedule, spatial & intensity augmentations).
- `unet3d.yaml`: 3D U-Net architecture parameters (`base_channels: 60`, `groups: 4`, AdamW $\text{LR}=3.0\times 10^{-4}$).
- `unetr.yaml`: UNETR architecture parameters (`embedding_dim: 384`, `num_layers: 6`, `num_heads: 8`, patch token size `16`, AdamW $\text{LR}=1.0\times 10^{-4}$).

---

## 📊 Experiment Tracking

Full experiment history is logged to MLflow under `outputs/mlruns/`.

To launch the MLflow UI:
```bash
uv run mlflow ui --backend-store-uri outputs/mlruns
```
Navigate to `http://127.0.0.1:5000`.

---

## 🏗️ Design Decisions

| Decision | Rationale |
| :--- | :--- |
| **MD5 Cross-Year Deduplication** | BraTS 2021 and 2024 overlap in patient samples. MD5 hashing raw NIfTI files guarantees zero data leakage into validation splits. |
| **Patch Size $128 \times 128 \times 128$** | Optimal 3D spatial context balance for brain MRI tumor boundaries fitting within GPU memory budgets. |
| **Physical Batch Size 2 & Grad Accum 4** | Physical batch size of 2 with 4 gradient accumulation steps achieves an effective batch size of 8 while avoiding CUDA OOM. |
| **GroupNorm (3D U-Net)** | Physical batch size of 2 makes `BatchNorm` unstable; `GroupNorm` (groups=4) ensures batch-size invariant normalization. |
| **Gradient Checkpointing (UNETR)** | Trades extra re-computation for memory reduction, dropping training VRAM from >22GB down to 7.03GB. |
| **Label 3 → 4 Remapping** | Resolves label schema mismatch between UPenn-GBM (label 3 for ET) and BraTS (label 4 for ET). |

---

## ⚠️ Limitations & Scope

1. **Benchmark Size & Single Cohort**: The out-of-distribution benchmark evaluated in this project consists exclusively of 30 cases from **UPenn-GBM** (all Glioblastoma). While it provides concrete OOD evidence, sample size variance is high.
2. **Descoped Datasets**: Secondary candidate datasets originally planned during project scoping (**TCGA-GBM/LGG** and **BraTS-SSA**) were not included in final benchmarking due to incomplete imaging modalities (e.g. missing T1 scans causing pipeline errors on TCGA) and data access constraints.
3. **GBM Scope**: UPenn-GBM consists solely of high-grade glioblastoma cases, meaning performance on low-grade gliomas (LGG) or non-GBM tumors was not evaluated.

---

## 🔒 Reproducibility

- Global random seed `42` is set deterministically across PyTorch, NumPy, Python, and MONAI.
- CuDNN deterministic flag enabled (`cudnn_benchmark: false`).
- UV dependency management ensures environment reproducibility.

---

*Documented from empirical logs, checkpoints, and benchmark reports on disk.*