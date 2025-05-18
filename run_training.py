import argparse
import torch
import datetime

from src.diffusion.model import LatentDiffusionModel
from src.diffusion.vae import VAE
from src.utils import login_huggingface, load_config
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path


RANDOM_SEED = 42
TEST_SIZE = 0.2


def get_args():
    parser = argparse.ArgumentParser(description="Train a latent diffusion model on Pong frames")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
    parser.add_argument("--model-name", type=str, default="latent_diffusion_unet", help="Base name (without extension) for the saved model")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory where the final model `.pth` will be saved")
    parser.add_argument("--hf-org", type=str, default="DiffusionArcade", help="Hugging Face organization to push the model to")

    return parser.parse_args()


def main():
    args = get_args()

    print('Logging into HuggingFace...')
    login_huggingface()

    print("Loading dataset (this might take a while)...")
    dataset = load_dataset("DiffusionArcade/Pong", split="train")
    split_datasets = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset = split_datasets["train"]
    val_dataset   = split_datasets["test"]

    print(f"Loading {args.config}...")
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(
        vae_name=config["training_latents"]["vae_name"],
        device=device,
        image_size=config['training_latents']['image_size']
    )

    print("Encoding images...")  
    train_dataset.set_transform(vae.encode_batch)
    val_dataset.set_transform(vae.encode_batch)

    batch_size = config["training_latents"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    print("Latent representations generated successfully!")

    # Initialize model
    model = LatentDiffusionModel(
        latent_channels=vae.latent_channels,
        latent_size=vae.latent_size,
        noise_scheduler=config["training_latents"]["scheduler"],
    )
    print("Model initialized successfully!")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['wandb']['run_name']}-{timestamp}"

    wandb_cfg = {
        "project": config["wandb"]["project"],
        "entity":  config["wandb"]["entity"],
        "name":    run_name,
        "config":  config
    }

    print("Training the model...")
    train_losses, val_losses = model.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config["training_latents"]["epochs"],
        lr=config["training_latents"]["learning_rate"],
        wandb_config=wandb_cfg,
    )

    print("Saving and pushing model...")
    model.save(
        output_dir=args.output_dir,
        hf_org=args.hf_org,
        model_name=args.model_name
    )
    print("Done!")


if __name__ == "__main__":
    main()

