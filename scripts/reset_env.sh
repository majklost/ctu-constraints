#!/bin/bash
#Run from project root: . ./reset_venv.sh
rm -rf ./.venv
echo "Virtual environment reset. Recreating..."
ml PyTorch-Lightning/2.5.5-foss-2025b-CUDA-12.9.1
echo "HPC modules loaded successfully."

python -m venv --system-site-packages ./.venv
echo "Virtual environment created at ./.venv"
source ./.venv/bin/activate
echo "Virtual environment activated."
pip install -e .
echo "Project dependencies installed in editable mode."