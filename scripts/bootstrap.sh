#!/bin/bash
# ODPlatform Bootstrap Script
# One-click environment setup

set -e

echo "=== ODPlatform Environment Setup ==="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n odp-gpu python=3.12 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate odp-gpu

# Install platform package
echo "Installing platform package..."
cd apps/platform
pip install -e ".[dev]"

# Download pretrained models
echo "Setting up models directory..."
mkdir -p ../../models/pretrained
mkdir -p ../../models/checkpoints

echo "=== Setup Complete ==="
echo "Activate environment with: conda activate odp-gpu"
