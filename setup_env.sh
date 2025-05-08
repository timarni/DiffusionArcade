#!/bin/bash

echo "Creating conda environment..."
conda create -n diffusion_arcade python=3.10 -y
conda activate diffusion_arcade

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing libraries..."
pip install -qq -U -r requirements.txt
echo "Done!"
