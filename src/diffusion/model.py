import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import wandb
import os
from datetime import datetime

from diffusers import UNet2DModel, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from tqdm import tqdm
from src.utils import push_model_to_hf, show_images
from src.diffusion.vae import VAE
from pathlib import Path


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
        noise_scheduler: str = 'DDPM',
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

        if noise_scheduler == 'DDPM':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )


    def compute_noise_weights(self, timesteps, gamma=5.0):
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
        snr = alpha_bar / (1 - alpha_bar)
        snr_t = snr.gather(-1, timesteps)
        weights = (snr_t + 1) / (snr_t + gamma)
        
        return weights
            

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 8,
        lr: float = 4e-4,
        wandb_config: dict = None,
        show_generations: bool = True,
    ):
        """Train the diffusion model over the provided dataloader"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        steps_per_epoch = len(train_dataloader)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            div_factor=25.0,
            final_div_factor=150.0,
            anneal_strategy="cos",
        )
        
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
                    self.noise_scheduler.config.num_train_timesteps,
                    (clean_images.size(0),),
                    device=self.device
                ).long()
                weights = self.compute_noise_weights(timesteps)

                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = self.model(noisy_images, timesteps).sample

                # Weight loss to allow the model to learn from images without much noise
                squared_diff = (noise_pred - noise)**2
                loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()

                loss.backward()
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

                if wandb_config is not None:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                global_step += 1

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)

            
            if show_generations:
                # Save samples after each epoch to see the quality of generations
                sample = self.generate(n_images=4, n_channels=1, num_inference_steps=1000)
                imgs = show_images(sample)
                wandb.log({"generated samples": wandb.Image(imgs)}, step=global_step)
            
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
                            self.noise_scheduler.config.num_train_timesteps,
                            (clean_images.size(0),),
                            device=self.device
                        ).long()
                        weights = self.compute_noise_weights(timesteps)

                        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
                        noise_pred = self.model(noisy_images, timesteps).sample

                        squared_diff = (noise_pred - noise) ** 2
                        loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()
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

            val_str = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {val_str} "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if wandb_config:
            wandb.finish()

        return train_losses, val_losses

    
    def generate(
        self,
        n_images: int = 8,
        n_channels: int = 3,
        num_inference_steps: int = 100,
    ):
        """Generate images from noise"""
        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Start from pure noise
        sample = torch.randn(n_images, n_channels, self.model.sample_size, self.model.sample_size).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                predicted_noise = self.model(sample, t).sample
                sample = self.noise_scheduler.step(predicted_noise, t, sample).prev_sample

        return sample

    
    def save(
        self,
        output_dir: str = 'models',
        hf_org: str = 'DiffusionArcade',
        model_name: str = 'diffusion_model'
    ):
        """Save model weights locally and push to Hugging Face Hub"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{model_name}.pth"
        torch.save(self.model.state_dict(), str(output_path))
        print(f"Model saved successfully to {output_path}!")
    
        hf_repo_id = f"{hf_org}/{model_name}"
        print(f"Pushing model to Hugging Face repo: {hf_repo_id}...")
        push_model_to_hf(str(output_path), hf_repo_id)

        
class LatentDiffusionModel:
    """
    Training and inference methods for a diffusion model in latent space.

    We add noise and denoise directly on latent representations from a VAE.
    """
    def __init__(
        self,
        device: torch.device = None,
        latent_channels: int = 4,
        latent_size: int = 16,
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
        noise_scheduler: str = "DDPM",
    ):
        """
        Initialize the UNet model and DDPM scheduler for latent diffusion.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.latent_channels = latent_channels
        self.latent_size = latent_size

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
        if noise_scheduler == "DDPM":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )


    def compute_noise_weights(self, timesteps, gamma=5.0):
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
        snr = alpha_bar / (1 - alpha_bar)
        snr_t = snr.gather(-1, timesteps)
        weights = (snr_t + 1) / (snr_t + gamma)
        
        return weights


    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 8,
        lr: float = 4e-4,
        wandb_config: dict = None,
        show_generations: bool = True,
    ):
        """Train the diffusion model on latent representations"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        steps_per_epoch = len(train_dataloader)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            div_factor=25.0,
            final_div_factor=150.0,
            anneal_strategy="cos",
        )
        
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
                latents = batch["latents"].to(self.device)

                noise = torch.randn_like(latents)

                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (latents.size(0),),
                    device=self.device
                ).long()
                weights = self.compute_noise_weights(timesteps)

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                noise_pred = self.model(noisy_latents, timesteps).sample

                # Weight loss to allow the model to learn from images without much noise
                squared_diff = (noise_pred - noise)**2
                loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

                if wandb_config:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                global_step += 1


            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)

            if show_generations:
                # Save samples after each epoch to see the quality of generations of this model
                vae = VAE(device=self.device, image_size=64)
                latents = self.generate_latents(n_samples=4, num_inference_steps=1000) # Shape should be [4, latent_channel, latent_size, latent_size]
                wandb.log({"latents": wandb.Image(latents)}, step=global_step)
                
                decoded_latents = vae.decode_latents(latents)
                imgs = show_images(decoded_latents)
                wandb.log({"generated samples": wandb.Image(imgs)}, step=global_step)
                del vae

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
                            self.noise_scheduler.config.num_train_timesteps,
                            (latents.size(0),),
                            device=self.device
                        ).long()
                        weights = self.compute_noise_weights(timesteps)

                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                        noise_pred = self.model(noisy_latents, timesteps).sample

                        squared_diff = (noise_pred - noise) ** 2
                        loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()
                        val_losses_epoch.append(loss.item())

                        global_step += 1

                avg_val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
                val_losses.append(avg_val_loss)

            # Log with wandb
            if wandb_config:
                log_data = {"epoch": epoch+1, "epoch/train_loss": avg_train_loss}
                if avg_val_loss is not None:
                    log_data["epoch/val_loss"] = avg_val_loss
                wandb.log(log_data, step=global_step)

            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss if avg_val_loss else 'N/A'} LR: {scheduler.get_last_lr()[0]:.2e}")

        if wandb_config:
            wandb.finish()

        return train_losses, val_losses


    def generate_latents(
        self,
        n_samples: int = 8,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """Generate latent samples from noise"""
        
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # start from Gaussian noise in latent space
        latents = torch.randn(n_samples, self.latent_channels, self.latent_size, self.latent_size).to(self.device)

        self.model.eval()
        
        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.model(latents, t).sample
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def save(
        self,
        output_dir: str = 'models',
        hf_org: str = 'DiffusionArcade',
        model_name: str = 'latent_diffusion_model'
    ):
        """Save model weights locally and push to Hugging Face Hub"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{model_name}.pth"
        torch.save(self.model.state_dict(), str(output_path))
        print(f"Model saved successfully to {output_path}!")
    
        hf_repo_id = f"{hf_org}/{model_name}"
        print(f"Pushing model to Hugging Face repo: {hf_repo_id}...")
        push_model_to_hf(str(output_path), hf_repo_id)



class ConditionedDiffusionModel:
    def __init__(
        self,
        image_size: int,
        in_channels: int = 5,
        out_channels: int = 1,
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
        noise_scheduler: str = 'DDPM',
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

        if noise_scheduler == 'DDPM':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )

    def compute_noise_weights(self, timesteps, gamma=5.0):
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
        snr = alpha_bar / (1 - alpha_bar)
        snr_t = snr.gather(-1, timesteps)
        weights = (snr_t + 1) / (snr_t + gamma)
        
        return weights
            
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 8,
        lr: float = 4e-4,
        wandb_config: dict = None,
        show_generations: bool = True,
    ):
        """Train the diffusion model over the provided dataloader"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        steps_per_epoch = len(train_dataloader)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            div_factor=25.0,
            final_div_factor=150.0,
            anneal_strategy="cos",
        )
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
            for (frames, actions), next_frames in loop:
                context_frames = frames.to(self.device)
                clean_next_frames = next_frames.to(self.device)

                noise = torch.randn_like(clean_next_frames)

                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (clean_next_frames.size(0),),
                    device=self.device
                ).long()                
                weights = self.compute_noise_weights(timesteps)

                noisy_next_frames = self.noise_scheduler.add_noise(clean_next_frames, noise, timesteps)
                context_frames_noisy = torch.cat([context_frames, noisy_next_frames.unsqueeze(1)], dim=1)
                
                noise_pred = self.model(context_frames_noisy, timesteps).sample # Shape: [B, 1, H, W]

                # Weight loss to allow the model to learn from images without much noise
                squared_diff = (noise_pred.squeeze(1) - noise)**2
                loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()

                loss.backward()
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

                if wandb_config is not None:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                global_step += 1

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            if show_generations and val_dataloader is not None:
                (frames, actions), _ = next(iter(val_dataloader)) # Shape: [B, C, H, W]

                B = frames.size(0)
                idx = torch.randperm(B)[:4]
                val_context = frames[idx].to(self.device)  # [4, C, H, W]
            
                with torch.no_grad():
                    pred_next = self.generate_next_frame(val_context, num_inference_steps=1000)  # Output: [4, H, W]

                C = val_context.shape[1]  # number of context frames
                
                imgs = []
                for i in range(4):
                    # get the C context frames + the generated one
                    ctx_i = val_context[i]                       # [C, H, W]
                    gen_i = pred_next[i].unsqueeze(0)          # [1, H, W]
                    row = torch.cat([ctx_i, gen_i], dim=0)     # [C+1, H, W]
                    for single in torch.unbind(row, dim=0):    # iterate over the C+1 frames
                        img = single * 0.5 + 0.5               # de-normalize
                        img_rgb = img.unsqueeze(0).repeat(3,1,1)  # → [3, H, W]
                        imgs.append(img_rgb)
                
                # Stack into a batch [4*(C+1), 3, H, W]
                batch = torch.stack(imgs, dim=0)
                
                # Make a 4×(C+1) grid
                grid = make_grid(batch, nrow=(C+1))
                
                wandb.log({
                    "val/generations": wandb.Image(grid)
                }, step=global_step)

            
            # Validation
            if val_dataloader is not None:
                self.model.eval()

                epoch_val_losses = []

                with torch.no_grad():
                    for (frames, actions), next_frames in val_dataloader:
                        context_frames = frames.to(self.device)
                        clean_next_frames = next_frames.to(self.device)

                        noise = torch.randn_like(clean_next_frames)

                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (clean_next_frames.size(0),),
                            device=self.device
                        ).long()
                        weights = self.compute_noise_weights(timesteps)

                        noisy_next_frames = self.noise_scheduler.add_noise(clean_next_frames, noise, timesteps)
                        context_frames_noisy = torch.cat([context_frames, noisy_next_frames.unsqueeze(1)], dim=1)

                        noise_pred = self.model(context_frames_noisy, timesteps).sample # Shape: [B, 1, H, W]

                        squared_diff = (noise_pred.squeeze(1) - noise) ** 2
                        loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()
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

            val_str = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {val_str} "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if wandb_config:
            wandb.finish()

        return train_losses, val_losses

    
    def generate_next_frame(
        self,
        context_frames: torch.Tensor,          # [B, C, H, W]
        num_inference_steps: int = 1000,
    ) -> torch.Tensor:
        """
        Generate the next frame conditioned on context_frames.
    
        Args:
            context_frames: a batch of previous frames, shape [B, C, H, W]
            num_inference_steps: number of diffusion steps
    
        Returns:
            Tensor of shape [B, H, W] with the predicted next frame.
        """
        # Prepare diffusion
        self.noise_scheduler.set_timesteps(num_inference_steps)
        self.model.eval()
    
        B, C, H, W = context_frames.shape

        # start from pure noise for the next frame channel
        sample = torch.randn(B, 1, H, W, device=self.device)
    
        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                # concatenate context and current sample -> [B, C+1, H, W]
                cond = torch.cat([context_frames.to(self.device), sample], dim=1)
                
                # predict noise on the “future” channel
                noise_pred = self.model(cond, t).sample  # [B, 1, H, W]
                
                # denoise one step
                sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
    
        return sample.squeeze(1) # Shape [B, H, W]


    def save(
        self,
        output_dir: str = 'models',
        hf_org: str = 'DiffusionArcade',
        model_name: str = 'diffusion_model'
    ):
        """Save model weights locally and push to Hugging Face Hub"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{model_name}.pth"
        torch.save(self.model.state_dict(), str(output_path))
        print(f"Model saved successfully to {output_path}!")
    
        hf_repo_id = f"{hf_org}/{model_name}"
        print(f"Pushing model to Hugging Face repo: {hf_repo_id}...")
        push_model_to_hf(str(output_path), hf_repo_id)



class ConditionedDiffusionModelWithAction(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int = 5,
        out_channels: int = 1,
        device: torch.device = None,
        timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        block_out_channels: tuple = (64, 128, 128, 256),
        layers_per_block: int = 2,
        action_embedding_size: int = 256,
        num_actions: int = 3,
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
        noise_scheduler: str = 'DDPM',
        unet_weights_path: str = None,  # NEW: path to your saved weights
    ):
        """
        Initialize the UNet model and DDIM scheduler.
        """
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.action_embedding_size = action_embedding_size

        self.action_embedding = torch.nn.Embedding(num_actions, action_embedding_size).to(self.device)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.model = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=action_embedding_size
        ).to(self.device)

        if unet_weights_path is not None:
            state_dict = torch.load(unet_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        if noise_scheduler == 'DDPM':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )

    def compute_noise_weights(self, timesteps, gamma=5.0):
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
        snr = alpha_bar / (1 - alpha_bar)
        snr_t = snr.gather(-1, timesteps)
        weights = (snr_t + 1) / (snr_t + gamma)
        
        return weights
            
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 8,
        lr: float = 4e-4,
        wandb_config: dict = None,
        show_generations: bool = True,
    ):
        """Train the diffusion model over the provided dataloader"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        steps_per_epoch = len(train_dataloader)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            div_factor=25.0,
            final_div_factor=150.0,
            anneal_strategy="cos",
        )
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
            
            for (frames, actions), next_frames in loop:
                context_frames = frames.to(self.device)
                clean_next_frames = next_frames.to(self.device)
                
                # Get the action embedding
                embedded_actions = self.action_embedding(actions.to(self.device))
                
                noise = torch.randn_like(clean_next_frames)

                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (clean_next_frames.size(0),),
                    device=self.device
                ).long()                
                weights = self.compute_noise_weights(timesteps)

                noisy_next_frames = self.noise_scheduler.add_noise(clean_next_frames, noise, timesteps)
                context_frames_noisy = torch.cat([context_frames, noisy_next_frames.unsqueeze(1)], dim=1)
                
                noise_pred = self.model(context_frames_noisy, timesteps, encoder_hidden_states=embedded_actions).sample # Shape: [B, 1, H, W]

                # Weight loss to allow the model to learn from images without much noise
                squared_diff = (noise_pred.squeeze(1) - noise)**2
                loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()

                loss.backward()
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())

                if wandb_config is not None:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                global_step += 1

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            if show_generations and val_dataloader is not None:
                (frames, actions), _ = next(iter(val_dataloader)) # Shape: [B, C, H, W]

                B = frames.size(0)
                idx = torch.randperm(B)[:4]
                val_context = frames[idx].to(self.device)  # [4, C, H, W]
                actions = actions[idx].to(self.device)
            
                with torch.no_grad():
                    pred_next = self.generate_next_frame(val_context, num_inference_steps=100, action=actions)  # Output: [4, H, W]

                C = val_context.shape[1]  # number of context frames
                
                imgs = []
                for i in range(4):
                    # get the C context frames + the generated one
                    ctx_i = val_context[i]                       # [C, H, W]
                    gen_i = pred_next[i].unsqueeze(0)          # [1, H, W]
                    row = torch.cat([ctx_i, gen_i], dim=0)     # [C+1, H, W]
                    for single in torch.unbind(row, dim=0):    # iterate over the C+1 frames
                        img = single * 0.5 + 0.5               # de-normalize
                        img_rgb = img.unsqueeze(0).repeat(3,1,1)  # → [3, H, W]
                        imgs.append(img_rgb)
                
                # Stack into a batch [4*(C+1), 3, H, W]
                batch = torch.stack(imgs, dim=0)
                
                # Make a 4×(C+1) grid
                grid = make_grid(batch, nrow=(C+1))
                
                wandb.log({
                    "val/generations": wandb.Image(grid)
                }, step=global_step)

            
            # Validation
            if val_dataloader is not None:
                self.model.eval()

                epoch_val_losses = []

                with torch.no_grad():
                    for (frames, actions), next_frames in val_dataloader:
                        context_frames = frames.to(self.device)
                        clean_next_frames = next_frames.to(self.device)

                        embedded_actions = self.action_embedding(actions.to(self.device))
                        
                        noise = torch.randn_like(clean_next_frames)

                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (clean_next_frames.size(0),),
                            device=self.device
                        ).long()
                        weights = self.compute_noise_weights(timesteps)

                        noisy_next_frames = self.noise_scheduler.add_noise(clean_next_frames, noise, timesteps)
                        context_frames_noisy = torch.cat([context_frames, noisy_next_frames.unsqueeze(1)], dim=1)

                        noise_pred = self.model(context_frames_noisy, timesteps, encoder_hidden_states=embedded_actions).sample # Shape: [B, 1, H, W]

                        squared_diff = (noise_pred.squeeze(1) - noise) ** 2
                        loss = (squared_diff * weights.view(-1, 1, 1, 1)).mean()
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

            val_str = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {val_str} "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            os.makedirs(f"./models/{self.timestamp}/checkpoints", exist_ok=True)
            
            if epoch % 10 == 9 and epoch > 0:
                torch.save(self.model.state_dict(), f"./models/{self.timestamp}/checkpoints/model_checkpoint_{self.timestamp}_{epoch+1}.pt", )
        
        os.makedirs(f"./models/{self.timestamp}/final", exist_ok=True)
        torch.save(self.model.state_dict(), f"./models/{self.timestamp}/final/model_{self.timestamp}.pt", )
        
        if wandb_config:
            wandb.finish()

        return train_losses, val_losses

    
    def generate_next_frame(
        self,
        context_frames: torch.Tensor,          # [B, C, H, W]
        action: torch.Tensor,
        num_inference_steps: int = 1000,
    ) -> torch.Tensor:
        """
        Generate the next frame conditioned on context_frames.
    
        Args:
            context_frames: a batch of previous frames, shape [B, C, H, W]
            num_inference_steps: number of diffusion steps
    
        Returns:
            Tensor of shape [B, H, W] with the predicted next frame.
        """
        # Prepare diffusion
        self.noise_scheduler.set_timesteps(num_inference_steps)
        self.model.eval()
    
        B, C, H, W = context_frames.shape

        # start from pure noise for the next frame channel
        sample = torch.randn(B, 1, H, W, device=self.device)

        action_embedding = self.action_embedding(action.to(self.device))
    
        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                # concatenate context and current sample -> [B, C+1, H, W]
                cond = torch.cat([context_frames.to(self.device), sample], dim=1)
                
                # predict noise on the “future” channel
                noise_pred = self.model(cond, t, encoder_hidden_states=action_embedding).sample  # [B, 1, H, W]
                
                # denoise one step
                sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
    
        return sample.squeeze(1) # Shape [B, H, W]

    def save(
        self,
        output_dir: str = 'models',
        hf_org: str = 'DiffusionArcade',
        model_name: str = 'diffusion_model'
    ):
        """Save model weights locally and push to Hugging Face Hub"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{model_name}.pth"
        torch.save(self.model.state_dict(), str(output_path))
        print(f"Model saved successfully to {output_path}!")
    
        hf_repo_id = f"{hf_org}/{model_name}"
        print(f"Pushing model to Hugging Face repo: {hf_repo_id}...")
        push_model_to_hf(str(output_path), hf_repo_id)

    @classmethod
    def from_pretrained(cls, path, **init_kwargs):
        # 1. Create an instance of the model
        model = cls(**init_kwargs)
        
        # 2. Load the state dict
        state_dict = torch.load(path, map_location='cpu')  # optional: map to 'cpu' or 'cuda'
        model.load_state_dict(state_dict)
    
        return model