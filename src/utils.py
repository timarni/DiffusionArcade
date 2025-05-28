import os
import numpy as np
import torchvision
import yaml
import datetime

from PIL import Image
from huggingface_hub import login, HfApi
from dotenv import load_dotenv

def login_huggingface():
    load_dotenv()

    token = os.environ.get("HF_ACCESS_TOKEN")

    if token is None:
        raise ValueError("HF_ACCESS_TOKEN environment variable not set. Please add it to your .env file.")
    login(token=token)


def load_config(config_path: str = "config.yaml"):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.asarray(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def get_formatted_run_name(run_name: str, model_type: str = "latent"):
    """Format a wandb run name to include date details"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_type}-{run_name}-{timestamp}"


def push_model_to_hf(model_path: str, repo_id: str):
    """
    Push a model file to Hugging Face Hub under the given repo ID

    Args:
        model_path: Path to the local model `.pth` file.
        repo_id: HF repo identifier (e.g. "org-name/model-name").
    """
    load_dotenv()

    token = os.environ.get("HF_ACCESS_TOKEN")
    if token is None:
        raise ValueError(
            "HF_ACCESS_TOKEN environment variable not set. Please add it to your .env file."
        )

    api = HfApi()
    
    # Create the repo on HF if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

    # Upload the file directly
    filename = os.path.basename(model_path)
    commit_msg = (
        f"Add model weights: {filename} @ "
        f"{datetime.datetime.utcnow().isoformat()}"
    )
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=filename,
        repo_id=repo_id,
        token=token,
        commit_message=commit_msg
    )

    print(f"Model pushed to Hugging Face: https://huggingface.co/{repo_id}/blob/main/{filename}")
