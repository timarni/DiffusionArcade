#!/usr/bin/env bash
# setup_mac.sh – one-shot installer for PyGame-Learning-Environment on macOS
# Run with:  bash setup_mac.sh

set -euo pipefail

echo "==> Checking for Homebrew …"
if ! command -v brew >/dev/null; then
  echo "Homebrew not found. Installing Homebrew …"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "==> Installing required system packages with Homebrew …"
brew install \
  git \
  sdl2 sdl2_image sdl2_mixer sdl2_ttf \
  portmidi libvorbis libogg libtiff libpng jpeg freetype

# ----------------------------------------------------------------------------- 
# Create a Python virtual environment
# ----------------------------------------------------------------------------- 
VENV_DIR="$HOME/ple-venv"
echo "==> Creating virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip and installing Python dependencies …"
pip install --upgrade pip
pip install numpy pillow pygame

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
echo "==> DONE! Activate your environment with:"
echo "        source ${VENV_DIR}/bin/activate"
echo ""
echo "If you are on a headless node, add these before running any code:"
echo "        export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=disk"
echo ""
echo "Smoke-test:  python -c 'from ple.games.pong import Pong; print(\"Pong OK\")'"
