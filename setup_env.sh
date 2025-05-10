#!/bin/bash

eval "$(conda shell.bash hook)"

echo "Creating conda environment..."
conda create -n diffusion_arcade python=3.10 -y
source activate diffusion_arcade

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing project and dependencies..."
pip install -e .

echo "Registering kernel..."
python -m ipykernel install --user --name diffusion_arcade --display-name "Python (diffusion_arcade)"

echo "Done!"
