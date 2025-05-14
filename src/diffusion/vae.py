import math
import torch

from torchvision import transforms
from diffusers import AutoencoderKL

class VAE:
    """A wrapper for a Hugging Face AutoencoderKL that provides encoding and decoding functions"""

    def __init__(self, vae_name: str, device: torch.device, image_size: int):
        super().__init__()
        self.device = device
        self.image_size = image_size

        # Load pretrained VAE and freeze parameters
        self.vae = AutoencoderKL.from_pretrained(vae_name).eval().to(self.device)
        self.vae.requires_grad_(False)

        self.scale = self.vae.config.scaling_factor
        self.latent_channels = self.vae.config.latent_channels
        self.latent_size = int(math.sqrt(self.vae.config.sample_size))

        print('The VAE was correctly instantiated!')

        # Preprocessing pipeline for input images
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)), # Resize image to be image_size x image_size
            transforms.ToTensor(), # Convert to tensor
            transforms.Normalize([0.5], [0.5]), # Normalize to be a tensor in [−1,1]
        ])

    def encode_batch(self, examples):
        """
        Encodes a batch of images into latent tensors.

        Args:
            examples (dict): A batch containing an "image" key with a list of PIL images.

        Returns:
            dict: A batch dict with key "latents" mapping to a tensor of shape [B, C, H, W].
        """
    
        # Convert to RGB and preprocess
        imgs = [self.preprocess(img.convert("RGB")) for img in examples["image"]]
        batch = torch.stack(imgs, dim=0).to(self.device)

        # Encode and sample latents
        with torch.no_grad():
            latents = self.vae.encode(batch).latent_dist.sample() * self.scale

        return {"latents": latents.cpu()}

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensors back into image tensors"""

        lat = latents.to(self.device) / self.scale
        with torch.no_grad():
            dec = self.vae.decode(lat).sample
        return dec.cpu()

