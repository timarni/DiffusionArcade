import torch
import torch.nn.functional as F
import wandb

from diffusers import UNet2DModel, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from tqdm import tqdm


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
        scheduler: str = 'DDPM',
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

        if scheduler == 'DDPM':
            self.scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.scheduler = DDIMScheduler(
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
        latents = torch.randn(n_samples, self.latent_channels, self.latent_size, self.latent_size).to(self.device)

        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.model(latents, t).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

class ConditionalDiffusion:
    
    def __init__(self,
        image_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        device: torch.device = None,
        timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        action_embedding_size: int = 8,
        num_actions: int = 3,
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
        scheduler: str = 'DDPM',
    ):

        """
        Initialize the UNet model and DDIM scheduler.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.action_embedding_size = action_embedding_size

        self.action_embedding = torch.nn.Embedding(num_actions, action_embedding_size).to(self.device)

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

        print("Action embedding size:\t", action_embedding_size)

        if scheduler == 'DDPM':
            self.scheduler = DDPMScheduler(
                num_train_timesteps=timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.scheduler = DDIMScheduler(
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
                
                # Embedding the action
                actions = torch.randint(0,3,(64,), device=self.device) # Batch size is hard coded for now !!!
                embedded_actions = self.action_embedding(actions)
                embedded_actions = embedded_actions.unsqueeze(1)
                
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (clean_images.size(0),),
                    device=self.device
                ).long()

                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=embedded_actions).sample
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

                        # Embedding the action
                        actions = torch.randint(0,3,(64,), device=self.device)
                        embedded_actions = self.action_embedding(actions)
                        embedded_actions = embedded_actions.unsqueeze(1)
                        # added_cond_kwargs = {"encoder_hidden_states": embedded_actions}
                        
                        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
                        noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=embedded_actions).sample
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
        action: int = 0,
    ):
        """Generate images from noise"""
        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)

        # Start from pure noise
        sample = torch.randn(n_images, n_channels, self.model.sample_size, self.model.sample_size).to(self.device)

        # Embed the action
        action = torch.tensor([action], device=self.device)                      
        action_embedding = self.action_embedding(action)
        action_embedding = action_embedding.unsqueeze(1) 

        for t in self.scheduler.timesteps:
            with torch.no_grad():
                residual = self.model(sample, t, encoder_hidden_states=action_embedding).sample
            sample = self.scheduler.step(residual, t, sample).prev_sample

        return sample

        
