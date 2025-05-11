# DiffusionArcade

A project that combines diffusion models with reinforcement learning for Pong game generation.

## Setup instructions

To set up the environment for DiffusionArcade, follow these steps:

1. Make the setup script executable:
   ```bash
   chmod +x setup_env.sh
   ```

2. Run the setup script to create and activate the conda environment, and install the necessary packages:
   ```bash
   source setup_env.sh
   ```

This will create a conda environment named `diffusion_arcade` with Python 3.10 and install all required packages defined in `pyproject.toml`. Ensure you have conda installed and available in your PATH.

## Usage

### Diffusion Models

Run the following notebook to test the diffusion model:

```bash
jupyter lab notebooks/diffusion_model.ipynb
```

# Using PLE to train a RL agent at Pong and recording the Game States

* Run `bash setup_ple_env.sh` to import all the necessary libraries (make sure to make it executable by `chmod +x setup_ple_env.sh`)
* Activate ple-venv `source /home/runai-home/ple-venv/bin/activate` (for EPFL RCP cluster, need to change path based on machine used)
* Inside ple-venv run `python3 qlearn_pong.py`
