#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --time=05:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out    # Output log file
#SBATCH --error=interactive_job.err     # Error log file

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

source "$HOME/miniconda3/etc/profile.d/conda.sh"  # Adjust path if different
conda activate diffusion_arcade
export WANDB_API_KEY=$WANDB && OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS run_training.py --config $CONFIG_FILE
