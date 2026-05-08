#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Creating conda environment..."
conda env create -f environment.yml

echo "[2/4] Activating..."
eval "$(conda shell.bash hook)"
conda activate cmb-anomaly

echo "[3/4] Installing project..."
pip install -e .

echo "[4/4] Done. Run: conda activate cmb-anomaly"
