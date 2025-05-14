import torch

from src.diffusion.model import LatentDiffusionModel
from src.diffusion.vae import VAE
from src.utils import login_huggingface, load_config

RANDOM_SEED=42
TEST_SIZE=0.2

def main():
    login_huggingface()

    print("Loading dataset...")
    dataset = load_dataset(
        "imagefolder",
        data_dir="../screens/",
        split="train"
    )
    split_datasets = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset = split_datasets["train"]
    val_dataset   = split_datasets["test"]

    print("Loading config.yaml...")
    config = load_config("config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_name = config["training_latents"]["vae_name"]
    image_size = config['training_latents']['image_size']
    
    vae = VAE(vae_name=vae_name, device=device, image_size=image_size)

    print("Encoding images...") 
    train_dataset.set_transform(vae.encode_batch)
    val_dataset.set_transform(vae.encode_batch)

    batch_size = config["training_latents"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Latent representations generated successfully!")

    latent_channels = vae.latent_channels
    latent_size = vae.latent_size
    scheduler = config["training_latents"]["scheduler"]
    epochs = config["training_latents"]["epochs"]
    lr = config["training_latents"]["learning_rate"]
    wandb_cfg = {
        "project": config["wandb"]["project"],
        "entity": config["wandb"]["entity"],
        "name": config['wandb']['run_name'],
        "config": config
    }
    
    model = LatentDiffusionModel(
        latent_channels=latent_channels,
        latent_size=latent_size,
        scheduler=scheduler, 
    )
    train_losses, val_losses = model.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        lr=lr,
        wandb_config=wandb_cfg,
    )
    
    torch.save(model.model.state_dict(), "latent_diffusion_unet.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
