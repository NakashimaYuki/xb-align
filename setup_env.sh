#!/bin/bash
# Linux/Mac shell script to setup XB-Align environment

set -e

echo "========================================"
echo "XB-Align Environment Setup"
echo "========================================"
echo

echo "Creating conda environment..."
conda create -n xb_align python=3.10 -y

echo
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xb_align

echo
echo "Installing core packages..."
pip install "numpy<2.0" pandas pyarrow openpyxl

echo
echo "Installing RDKit..."
conda install -c conda-forge rdkit -y

echo
echo "Installing PyTorch (CPU)..."
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

echo
echo "Installing PyTorch Geometric..."
pip install torch-geometric

echo
echo "Installing utilities..."
pip install scikit-learn tqdm pyyaml pytest

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "To activate the environment:"
echo "  conda activate xb_align"
echo
echo "To run tests:"
echo "  pytest"
echo
echo "To run M1 pipeline:"
echo "  python run_m1_pipeline.py"
echo
