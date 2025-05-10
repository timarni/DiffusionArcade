#!/usr/bin/env bash
# setup.sh  –  one‑shot installer for PyGame‑Learning‑Environment on Ubuntu ≥ 20.04
# Run with:  bash setup_ple_env.sh
# If you are on a headless cluster you can add:  SDL_VIDEODRIVER=dummy ./setup_ple_env.sh

set -euo pipefail

ENV_NAME="ple_env"
echo "==> Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install dependencies via conda or pip
echo "==> Installing Python dependencies"
conda install -y pip numpy pillow
pip install pygame  # Conda version may be outdated

# Clone and install PLE in editable mode
PLE_DIR="$HOME/PyGame-Learning-Environment"
if [ ! -d "$PLE_DIR" ]; then
  echo "==> Cloning PLE repo"
  git clone https://github.com/ntasfi/PyGame-Learning-Environment.git "$PLE_DIR"
fi

echo "==> Installing PLE (editable) …"
pip install -e "$PLE_DIR"

# ----------------------------------------------------------------------------- 
echo ""
echo "==> DONE! To activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "If you're on a headless node (no GUI), before running code, do:"
echo "    export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=disk"
echo ""
echo "Smoke‑test:  python -c 'from ple.games.pong import Pong; print(\"Pong OK\")'"
