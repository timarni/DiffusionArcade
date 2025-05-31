#!/bin/bash

eval "$(conda shell.bash hook)"

echo "Creating conda environment..."
conda create -n diffusion_arcade python=3.10 -y
source activate diffusion_arcade

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing project and dependencies..."
pip install -e .

# Clone and install PLE in editable mode
PLE_DIR="$HOME/PyGame-Learning-Environment"
if [ ! -d "$PLE_DIR" ]; then
  echo "==> Cloning PLE repo"
  git clone https://github.com/ntasfi/PyGame-Learning-Environment.git "$PLE_DIR"
fi

echo "==> Installing PLE (editable) …"
pip install -e "$PLE_DIR"

echo "Registering kernel..."
python -m ipykernel install --user --name diffusion_arcade --display-name "Python (diffusion_arcade)"

echo "Done!"
