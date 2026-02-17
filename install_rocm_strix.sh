#!/bin/bash
# Installation script for AMD Strix Halo (gfx1151) with ROCm 7.1

echo "Installing dependencies for AMD Strix Halo (gfx1151)..."

# Install dependencies using the custom requirements file
pip install -r requirements_rocm.txt

if [ $? -eq 0 ]; then
    echo "Installation complete."
    echo "Note: liger-kernel is skipped in requirements_rocm.txt to avoid build issues."
    echo "Note: flash-attn is also skipped; PyTorch SDPA will be used instead."
else
    echo "Installation failed."
    exit 1
fi
