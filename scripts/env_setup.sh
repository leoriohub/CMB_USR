#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cmb-anomaly"
ENV_FILE="environment.yml"

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if conda env list | grep -q "^$ENV_NAME "; then
    echo "[update] $ENV_NAME already exists — updating dependencies..."
    conda env update -f "$ENV_FILE" --prune
else
    echo "[create] Creating $ENV_NAME from $ENV_FILE..."
    conda env create -f "$ENV_FILE"
fi

echo "[kernel] Installing ipykernel for Jupyter..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "[verify] Running verification..."
python scripts/verify_env.py

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
