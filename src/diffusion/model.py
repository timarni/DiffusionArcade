import os
import torch
import torch.nn.functional as F
import wandb
import yaml

from diffusers import UNet2DModel, DDIMScheduler, DDPMScheduler
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


class DiffusionModel:
    """
    Training and inference methods for a diffusion model using UNet and DDIM scheduler.

    Forward process: add noise to images across timesteps according to a variance schedule.
    Reverse process: denoise starting from Gaussian noise to reconstruct samples.
    """
    def __init__(
        self,
        image_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
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
        Initialize the UNet model and DDIM scheduler.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        ).to(self.device)

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
        """Train the diffusion model over the provided dataloader"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        global_step = 0

        if wandb_config is not None:
            wandb.init(**wandb_config)

        for epoch in range(epochs):
            self.model.train()

            epoch_train_losses = []

            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in loop:
                clean_images = batch["image"].to(self.device)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (clean_images.size(0),),
                    device=self.device
                ).long()

                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = self.model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

                if wandb_config is not None:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                global_step += 1

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)

            # Validation
            if val_dataloader is not None:
                self.model.eval()

                epoch_val_losses = []

                with torch.no_grad():
                    for batch in val_dataloader:
                        clean_images = batch["image"].to(self.device)
                        noise = torch.randn_like(clean_images)
                        timesteps = torch.randint(
                            0,
                            self.scheduler.num_train_timesteps,
                            (clean_images.size(0),),
                            device=self.device
                        ).long()

                        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
                        noise_pred = self.model(noisy_images, timesteps).sample
                        loss = F.mse_loss(noise_pred, noise)
                        epoch_val_losses.append(loss.item())

                        global_step += 1

                avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
                val_losses.append(avg_val_loss)
            else:
                avg_val_loss = None

            if wandb_config is not None:
                log_data = {"epoch": epoch + 1, "epoch/train_loss": avg_train_loss}
                if avg_val_loss is not None:
                    log_data["epoch/val_loss"] = avg_val_loss
                wandb.log(log_data, step=global_step)

            # Print summary
            val_str = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {val_str}"
            )

        return train_losses, val_losses

    def generate(
        self,
        n_images: int = 8,
        n_channels: int = 3,
        num_inference_steps: int = 100,
    ):
        """Generate images from noise"""
        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)

        # Start from pure noise
        sample = torch.randn(n_images, n_channels, self.model.sample_size, self.model.sample_size).to(self.device)

        for t in self.scheduler.timesteps:
            with torch.no_grad():
                residual = self.model(sample, t).sample
            sample = self.scheduler.step(residual, t, sample).prev_sample

        return sample
