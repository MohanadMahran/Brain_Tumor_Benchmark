# 🧠 3D Brain Tumor Segmentation Benchmark: 3D U-Net vs UNETR

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-2d335c.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, highly-optimized, and fair benchmarking framework for comparing 3D Convolutional Neural Networks (3D U-Net) and Vision Transformers (UNETR) on the complex task of brain tumor segmentation using multi-modal MRI scans. 

This project trains models on a deduplicated composite dataset derived from **BraTS 2021** and **BraTS 2024 Adult Glioma**, and evaluates them rigorously on independent, held-out benchmark datasets: **TCGA-GBM/LGG** and **BraTS-SSA**.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Project Structure](#-project-structure)
- [Running the Pipeline](#-running-the-pipeline)
- [Detailed Usage](#-detailed-usage)
- [Configuration](#-configuration)
- [Experiment Tracking](#-experiment-tracking)
- [Design Decisions](#-design-decisions)
- [Reproducibility](#-reproducibility)

---

## 🌟 Overview

The goal of this benchmark is to provide a meticulously controlled environment to compare CNN and Transformer architectures for 3D medical image segmentation. 

### Evaluated Architectures:
1. **3D U-Net**: The gold-standard CNN architecture. Configured with ~19M parameters, using `GroupNorm` (since batch size is 1), and a purely convolutional encoder-decoder structure.
2. **UNETR (UNet Transformers)**: A state-of-the-art vision transformer. Configured with ~19M parameters, utilizing `LayerNorm`, a ViT-based encoder, and a CNN-based decoder. Incorporates gradient checkpointing to fit within strict VRAM constraints.

Both models process 4-channel input (T1, T1Gd, T2, FLAIR) and predict 3 sub-regions:
- Enhancing Tumor (ET)
- Tumor Core (TC)
- Whole Tumor (WT)

---

## ✨ Key Features

- **Fair Comparison**: Parameter-matched models (~19M each) trained under identical conditions, augmentations, and optimizers.
- **VRAM Optimized**: Designed to train 3D models on consumer GPUs (8GB VRAM) using FP16 Mixed Precision, Gradient Accumulation, and Gradient Checkpointing.
- **Robust Data Handling**: Includes MD5-based cross-year deduplication to prevent data leakage between BraTS 2021 and BraTS 2024.
- **Advanced Metrics**: Implements custom volumetric Dice coefficient and 95th percentile Hausdorff Distance (HD95) for edge-case resilient 3D evaluation.
- **Comprehensive Tracking**: Fully integrated with **MLflow** for hyperparameter tracking, metric logging, and artifact management.

---

## 💻 Hardware Requirements

| Component | Minimum Specification | Recommended Specification |
|-----------|-----------------------|---------------------------|
| **GPU** | NVIDIA RTX 2080 / 3060 / 4060 | NVIDIA RTX 3090 / 4090 |
| **VRAM** | 8 GB | 24 GB |
| **CUDA** | 11.8+ (Tested on 12.1+) | 12.1+ |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 150 GB free (SSD/NVMe) | 500 GB NVMe |
| **OS** | Linux (Ubuntu 20.04+), WSL2 | Linux (Ubuntu 22.04+) |

---

## 🛠 Installation

This project utilizes [Astral UV](https://github.com/astral-sh/uv) for blazing-fast dependency management and resolution.

1. **Install UV:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd brain_tumor_benchmark
   ```

3. **Install Dependencies:**
   ```bash
   uv sync
   ```
   *Note: The `uv.lock` file ensures that exact dependency versions are installed for deterministic execution.*

---

## 📂 Data Preparation

The pipeline expects datasets to be downloaded and extracted into the `data/raw/` directory following a specific structure.

```text
data/raw/
├── brats2021/          # BraTS 2021 (from Kaggle)
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
├── brats2024/          # BraTS 2024 Adult Glioma (from Kaggle)
│   └── ...
├── tcga/               # TCGA-GBM + TCGA-LGG (from TCIA)
│   └── ...
└── brats_ssa/          # BraTS-SSA Sub-Saharan Africa (from Kaggle/Synapse)
    └── ...
```

---

## 🚀 Running the Pipeline

To execute the entire end-to-end pipeline automatically (deduplication, training both models, and running both benchmarks), simply run:

```bash
bash run_all.sh
```

### Pipeline Sequence:
1. **Deduplication (`verify_deduplicate.py`)**: Checks for overlapping patients and generates safe train/val splits.
2. **Train 3D U-Net (`train_unet3d.py`)**: Trains the U-Net model.
3. **Train UNETR (`train_unetr.py`)**: Trains the UNETR model.
4. **Benchmark TCGA (`benchmark.py`)**: Evaluates both models on the TCGA dataset.
5. **Benchmark BraTS-SSA (`benchmark.py`)**: Evaluates both models on the BraTS-SSA dataset.
6. **Generate Reports**: Outputs comparison metrics to CSV files.

---

## 🔍 Detailed Usage

### 1. Training a Specific Model
You can run individual training scripts explicitly.

**For 3D U-Net:**
```bash
uv run python scripts/train_unet3d.py \
    --config configs/unet3d.yaml \
    --base_config configs/base.yaml \
    --output_dir outputs/models/unet3d \
    --log_dir outputs/logs/unet3d
```

**For UNETR:**
```bash
uv run python scripts/train_unetr.py \
    --config configs/unetr.yaml \
    --base_config configs/base.yaml \
    --output_dir outputs/models/unetr \
    --log_dir outputs/logs/unetr
```

### 2. Running Individual Benchmarks
After training, evaluate the models on specific datasets:

**TCGA Benchmark:**
```bash
uv run python scripts/benchmark.py \
    --benchmark tcga \
    --data_dir data/raw/tcga \
    --unet_checkpoint outputs/models/unet3d/best_checkpoint.pth \
    --unetr_checkpoint outputs/models/unetr/best_checkpoint.pth \
    --output_dir outputs/reports/tcga \
    --base_config configs/base.yaml
```

---

## ⚙️ Configuration

Hyperparameters, model settings, and augmentation parameters are decoupled into YAML files within the `configs/` directory.

- `base.yaml`: Shared configuration. Defines data directories, global seed (42), batch size, learning rate schedule, spatial augmentations (rotations, flips), and intensity augmentations (noise, shift).
- `unet3d.yaml`: Specific settings for the 3D U-Net (channels, strides, normalization).
- `unetr.yaml`: Specific settings for UNETR (hidden size, MLP dimensions, number of heads).

---

## 📊 Experiment Tracking

This project natively integrates **MLflow** for robust experiment tracking.

To view training curves, memory usage, hyperparameters, and evaluation metrics:
```bash
uv run mlflow ui --backend-store-uri outputs/mlruns
```
Open your browser to `http://127.0.0.1:5000`.

**Tracked Metrics:**
- Train/Val Loss (DiceCE)
- Validation Dice (ET, TC, WT, Mean)
- Validation HD95
- Peak VRAM Allocation
- Epoch durations

---

## 🏗️ Design Decisions

| Decision | Rationale |
|----------|-----------|
| **MD5 Deduplication** | BraTS 2021 and 2024 share overlapping patients. MD5 hashing raw `.nii.gz` files ensures no data leakage between train/validation splits. |
| **Patch Size 96³** | The largest spatial context (train patch size) that reliably fits into 8GB VRAM for both the U-Net and UNETR architectures. |
| **Batch Size & Grad Accumulation** | Physical batch size is set to `1` (memory limit). Gradient accumulation of `2` steps provides an effective batch size of `2`. |
| **GroupNorm (U-Net)** | Since batch size is 1, standard `BatchNorm` is highly unstable. `GroupNorm` is independent of batch size. |
| **Gradient Checkpointing** | Essential for UNETR to fit the Transformer encoder into an 8GB VRAM buffer by trading compute for memory. |
| **Mixed Precision (FP16)** | Reduces VRAM footprint by ~50% and speeds up compute via Tensor Cores. |
| **Cosine Annealing** | A robust learning rate scheduler that works excellently for both CNN and Transformer-based models. |
| **Custom 4-Channel Adapter** | Standard Vision Transformers expect 3-channel RGB. An adapter layer handles the 4-channel MRI inputs (T1, T1Gd, T2, FLAIR). |

---

## 🔒 Reproducibility

Ensuring identical, deterministic runs is a core tenet of this benchmark:
- A global random seed (`42`) is injected across Python, NumPy, PyTorch, and MONAI.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` are enforced.
- Deduplication manifests are explicitly saved and loaded to guarantee exact data splits.
- Dependency drift is prevented via the `uv.lock` lockfile.

---

*Built for the accurate and fair comparison of deep learning architectures in neuro-oncology.*
