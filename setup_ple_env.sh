#!/usr/bin/env bash
# setup.sh  –  one‑shot installer for PyGame‑Learning‑Environment on Ubuntu ≥ 20.04
# Run with:  bash setup_ple_env.sh
# If you are on a headless cluster you can add:  SDL_VIDEODRIVER=dummy ./setup_ple_env.sh

set -euo pipefail
echo "==> Updating APT and installing system libraries …"
sudo apt-get update

# --- SDL2 + codecs + build toolchain -----------------------------------------
sudo apt-get install -y \
  git build-essential cython3 \
  python3 python3-dev python3-pip python3-venv python3-setuptools python3-virtualenv \
  libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
  libfreetype6-dev libportmidi-dev libjpeg-dev libtiff5-dev \
  libavcodec-dev libavformat-dev libswscale-dev libsmpeg-dev \
  libx11-6 libx11-dev

# ----------------------------------------------------------------------------- 
# Create an isolated virtual environment (modify the path if you prefer)
# ----------------------------------------------------------------------------- 
VENV_DIR="$HOME/ple-venv"
echo "==> Creating virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip and installing Python‑level deps …"
pip install --upgrade pip
pip install numpy pillow pygame              # PLE soft deps + PyGame 2
pip install matplotlib
pip install tqdm
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jsonargparse 

# ----------------------------------------------------------------------------- 
# Clone and install PLE in editable mode
# ----------------------------------------------------------------------------- 
PLE_DIR="$HOME/PyGame-Learning-Environment"
if [ ! -d "${PLE_DIR}" ]; then
  echo "==> Cloning PLE repository …"
  git clone https://github.com/ntasfi/PyGame-Learning-Environment.git "${PLE_DIR}"
fi

echo "==> Installing PLE (editable) …"
pip install -e "${PLE_DIR}"

# ----------------------------------------------------------------------------- 
echo "==> DONE!  Activate your environment with:"
echo "        source ${VENV_DIR}/bin/activate"
echo ""
echo "If you are on a headless node, add these before running any code:"
echo "        export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=disk"
echo ""
echo "Smoke‑test:  python -c 'from ple.games.pong import Pong; print(\"Pong OK\")'"
