import os
import argparse
import torch
import torch.nn.functional as F

from diffusers import UNet2DModel, DDIMScheduler
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm


def login_huggingface():
    load_dotenv()

    token = os.environ.get("HF_ACCESS_TOKEN")

    if token is None:
        raise ValueError("HF_ACCESS_TOKEN environment variable not set. Please add it to your .env file.")
    login(token=token)


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
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        ).to(self.device)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule=beta_schedule
        )

    def train(
        self,
        train_dataloader,
        epochs: int = 8,
        lr: float = 4e-4
    ):
        """Train the diffusion model over the provided dataloader"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in loop:
                clean_images = batch["images"].to(self.device)

                # Sample noise to add to the images
                noise = torch.randn_like(clean_images)
                bs = clean_images.size(0)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (bs,),
                    device=self.device
                ).long()

                # Add noise to the clean images at the sampled timestep
                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

                # Predict the noise
                noise_pred = self.model(noisy_images, timesteps).sample

                # Compute the loss
                loss = F.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

            if (epoch + 1) % 2 == 0:
                avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
                print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

        return losses

    def generate(
        self,
        batch_size: int = 8,
        n_channels: int = 3,
        num_inference_steps: int = 100,
    ):
        """Generate images from noise"""

        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)

        # Start from pure noise
        sample = torch.randn(batch_size, n_channels, self.model.image_size, self.model.image_size).to(self.device)

        for t in self.scheduler.timesteps:
            with torch.no_grad():
                residual = self.model(sample, t).sample
            sample = self.scheduler.step(residual, t, sample).prev_sample

        return sample
