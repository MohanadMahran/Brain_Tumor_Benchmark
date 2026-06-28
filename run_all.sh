#!/bin/bash
set -e

echo "============================================================"
echo " Brain Tumor Segmentation Benchmark — 3D U-Net vs UNETR"
echo "============================================================"
echo ""

# Activate virtual environment created by UV
source .venv/bin/activate
export PYTHONHASHSEED=42
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directories
mkdir -p outputs/models/unet3d
mkdir -p outputs/models/unetr
mkdir -p outputs/logs/unet3d
mkdir -p outputs/logs/unetr
mkdir -p outputs/mlruns
mkdir -p outputs/reports/tcga
mkdir -p outputs/reports/ssa

echo "[1/6] Deduplication and manifest generation..."
uv run python scripts/verify_deduplicate.py \
  --brats2021_dir data/raw/brats2021 \
  --brats2024_dir data/raw/brats2024 \
  --output_dir outputs/logs/
if [ $? -ne 0 ]; then
    echo "ERROR: Deduplication failed."
    exit 1
fi
echo "      Deduplication complete."
echo ""

echo "[2/6] Training 3D U-Net..."
uv run python scripts/train_unet3d.py \
  --config configs/unet3d.yaml \
  --base_config configs/base.yaml \
  --output_dir outputs/models/unet3d \
  --log_dir outputs/logs/unet3d
if [ $? -ne 0 ]; then
    echo "ERROR: U-Net training failed."
    exit 1
fi
echo "      U-Net training complete."
echo ""

echo "[3/6] Training UNETR..."
uv run python scripts/train_unetr.py \
  --config configs/unetr.yaml \
  --base_config configs/base.yaml \
  --output_dir outputs/models/unetr \
  --log_dir outputs/logs/unetr
if [ $? -ne 0 ]; then
    echo "ERROR: UNETR training failed."
    exit 1
fi
echo "      UNETR training complete."
echo ""

echo "[4/6] Benchmarking on TCGA-GBM/LGG..."
uv run python scripts/benchmark.py \
  --benchmark tcga \
  --unet_checkpoint outputs/models/unet3d/best_checkpoint.pth \
  --unetr_checkpoint outputs/models/unetr/best_checkpoint.pth \
  --data_dir data/raw/tcga \
  --output_dir outputs/reports/tcga
if [ $? -ne 0 ]; then
    echo "WARNING: TCGA benchmark had errors (check logs)."
fi
echo "      TCGA benchmark complete."
echo ""

echo "[5/6] Benchmarking on BraTS-SSA..."
uv run python scripts/benchmark.py \
  --benchmark ssa \
  --unet_checkpoint outputs/models/unet3d/best_checkpoint.pth \
  --unetr_checkpoint outputs/models/unetr/best_checkpoint.pth \
  --data_dir data/raw/brats_ssa \
  --output_dir outputs/reports/ssa
if [ $? -ne 0 ]; then
    echo "WARNING: SSA benchmark had errors (check logs)."
fi
echo "      BraTS-SSA benchmark complete."
echo ""

echo "[6/6] Generating final comparison report..."
uv run python evaluation/report.py \
  --reports_dir outputs/reports \
  --output outputs/reports/final_comparison.csv
echo "      Final report generated."
echo ""

echo "============================================================"
echo " ALL DONE."
echo " View results: mlflow ui --backend-store-uri outputs/mlruns"
echo "============================================================"
