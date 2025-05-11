import os
import torch
import torch.nn.functional as F
import wandb
import yaml
import math

from diffusers import UNet2DModel, DDPMScheduler, AutoencoderKL
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm


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


class LatentDiffusionModel:
    """
    Training and inference methods for a diffusion model in latent space.

    We add noise and denoise directly on latent representations from a VAE.
    """
    def __init__(
        self,
        vae: AutoencoderKL,
        device: torch.device = None,
        timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        block_out_channels: tuple = (64, 128, 128, 256),
        layers_per_block: int = 2,
        down_block_types: tuple = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: tuple = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ):
        """
        Initialize the UNet model and DDPM scheduler for latent diffusion.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # VAE for encoding/decoding
        self.vae = vae.eval().to(self.device)
        self.vae.requires_grad_(False)
        self.scale = self.vae.config.scaling_factor
        
        # latent shape
        latent_channels = self.vae.config.latent_channels
        latent_size = int(math.sqrt(self.vae.config.sample_size))

        # UNet in latent space
        self.model = UNet2DModel(
            sample_size=latent_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        ).to(self.device)

        # scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule=beta_schedule
        )

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 8,
        lr: float = 4e-4,
        wandb_config: dict = None,
    ):
        """Train the diffusion model on latent representations"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        global_step = 0

        if wandb_config:
            wandb.init(**wandb_config)

        for epoch in range(epochs):
            self.model.train()
            epoch_train_losses = []
            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in loop:
                # Get latent vectors from batch
                latents = batch["latents"].to(self.device)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (latents.size(0),),
                    device=self.device
                ).long()

                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                noise_pred = self.model(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

                if wandb_config:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                global_step += 1

            train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))

            # Validation
            avg_val_loss = None
            if val_dataloader:
                self.model.eval()
                val_losses_epoch = []
                with torch.no_grad():
                    for batch in val_dataloader:
                        latents = batch["latents"].to(self.device)
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(
                            0,
                            self.scheduler.num_train_timesteps,
                            (latents.size(0),),
                            device=self.device
                        ).long()
                        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                        noise_pred = self.model(noisy_latents, timesteps).sample
                        val_losses_epoch.append(F.mse_loss(noise_pred, noise).item())
                        global_step += 1

                avg_val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
                val_losses.append(avg_val_loss)

            # Log with wandb
            if wandb_config:
                log_data = {"epoch": epoch+1, "epoch/train_loss": train_losses[-1]}
                if avg_val_loss is not None:
                    log_data["epoch/val_loss"] = avg_val_loss
                wandb.log(log_data, step=global_step)

            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_losses[-1]:.4f} Val Loss: {avg_val_loss if avg_val_loss else 'N/A'}")

        return train_losses, val_losses

    def generate_latents(
        self,
        n_samples: int = 8,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """Generate latent samples from noise"""
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # start from Gaussian noise in latent space
        latents = torch.randn(
            n_samples,
            self.vae.config.latent_channels,
            self.vae.config.sample_size,
            self.vae.config.sample_size,
        ).to(self.device)

        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.model(latents, t).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to image tensor"""
        lat_scaled = latents / self.scale
        with torch.no_grad():
            imgs = self.vae.decode(lat_scaled).sample
        # imgs in [-1,1], convert to [0,1]
        imgs = (imgs.clamp(-1,1) + 1) / 2
        return imgs.cpu()

    def generate_images(
        self,
        n_samples: int = 8,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """Generate images by sampling latents then decoding"""
        lat = self.generate_latents(n_samples, num_inference_steps)
        imgs = self.decode_latents(lat)
        return imgs