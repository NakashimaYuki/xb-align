@echo off
REM Windows batch script to setup XB-Align environment

echo ========================================
echo XB-Align Environment Setup
echo ========================================
echo.

echo Creating conda environment...
call conda create -n xb_align python=3.10 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    exit /b 1
)

echo.
echo Activating environment...
call conda activate xb_align

echo.
echo Installing core packages...
pip install "numpy<2.0" pandas pyarrow openpyxl
if errorlevel 1 (
    echo ERROR: Failed to install core packages
    exit /b 1
)

echo.
echo Installing RDKit...
call conda install -c conda-forge rdkit -y
if errorlevel 1 (
    echo ERROR: Failed to install RDKit
    exit /b 1
)

echo.
echo Installing PyTorch (CPU)...
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)

echo.
echo Installing PyTorch Geometric...
pip install torch-geometric
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch Geometric
    exit /b 1
)

echo.
echo Installing utilities...
pip install scikit-learn tqdm pyyaml pytest
if errorlevel 1 (
    echo ERROR: Failed to install utilities
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To activate the environment:
echo   conda activate xb_align
echo.
echo To run tests:
echo   pytest
echo.
echo To run M1 pipeline:
echo   python run_m1_pipeline.py
echo.
