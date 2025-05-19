import argparse
import torch
import datetime

from src.diffusion.model import ConditionedDiffusionModel
from src.utils import login_huggingface, load_config
from src.diffusion.data_preprocessing import ContextFrameDataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets import load_dataset
from pathlib import Path


RANDOM_SEED = 42
TEST_SIZE = 0.2


def get_args():
    parser = argparse.ArgumentParser(description="Train a latent diffusion model on Pong frames")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
    parser.add_argument("--model-name", type=str, default="conditioned_diffusion_unet_60_epochs", help="Base name (without extension) for the saved model")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory where the final model `.pth` will be saved")
    parser.add_argument("--hf-org", type=str, default="DiffusionArcade", help="Hugging Face organization to push the model to")

    return parser.parse_args()


def main():
    args = get_args()

    print('Logging into HuggingFace...')
    login_huggingface()

    print("Loading dataset (this might take a while)...")
    raw_dataset = load_dataset("DiffusionArcade/Pong", split="train")

    print(f"Loading {args.config}...")
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_size = config['training_conditioned_images']['image_size']
    in_channels=config['training_conditioned_images']['in_channels']
    context_length = in_channels - 1
    
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Normalized between [-1, 1]
    ])
    
    context_dataset = ContextFrameDataset(raw_dataset, context_length=context_length, step=2, transform=preprocess)

    batch_size = config['training_conditioned_images']['batch_size']

    train_ds, val_ds = context_dataset.train_val_split(val_ratio=TEST_SIZE, random_seed=RANDOM_SEED)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    image_size=config['training_conditioned_images']['image_size']
    in_channels=config['training_conditioned_images']['in_channels']
    out_channels=config['training_conditioned_images']['out_channels']
    timesteps = config['training_conditioned_images']['timesteps']
    beta_schedule=config['training_conditioned_images']['beta_schedule']
    
    model = ConditionedDiffusionModel(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
    )
    print("Model initialized successfully!")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"conditioned-{config['wandb']['run_name']}-{timestamp}"
    wandb_cfg = {
        "project": config["wandb"]["project"],
        "entity":  config["wandb"]["entity"],
        "name":    run_name,
        "config":  config
    }

    epochs = config["training_conditioned_images"]["epochs"]
    lr = config["training_conditioned_images"]["learning_rate"]

    print("Training the model...")
    train_losses, val_losses = model.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        lr=lr,
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

