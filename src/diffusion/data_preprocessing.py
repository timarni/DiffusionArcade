import torch
import pandas as pd

from torch.utils.data import Dataset, random_split
from torchvision import transforms


class ContextFrameDataset(Dataset):
    """
    PyTorch Dataset for context-based frame prediction.

    Each sample is a tuple: ((context_frames, context_actions), ground_truth)

    Args:
        dataset: Iterable with items having keys ['image', 'action', 'episode', 'step']
        context_length (int): Number of context frames to use as input.
        step (int): Stride between frames (default=2).
        transform (callable, optional): Transform to apply to each image.
    """
    def __init__(self, dataset, context_length, step=2, transform_context=None, transform_gt=None):
        self.dataset = dataset
        self.context_length = context_length
        self.step = step
        self.transform_context = transform_context
        self.transform_gt = transform_gt
        self.samples = []

        # Build index mapping per episode
        df = dataset.to_pandas()

        grouped = df.groupby("episode")

        for episode, group in grouped:
            group = group.sort_values("step").reset_index()
            max_i = len(group)
            min_i = context_length * step
            for i in range(min_i, max_i, step):
                idxs = [group.loc[i - step * (context_length - j), "index"]
                        for j in range(context_length)]
                gt_idx = int(group.loc[i, "index"])
                self.samples.append((idxs, gt_idx))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        idxs, gt_idx = self.samples[idx]

        # Load raw images and actions
        context_frames = [self.dataset[int(i)]["image"] for i in idxs]
        context_actions = [self.dataset[int(i)]["action"] for i in idxs]
        ground_truth = self.dataset[int(gt_idx)]["image"]

        # Apply transforms if provided (each frame: 1xHxW)
        if self.transform_context:
            context_frames = [self.transform_context(img) for img in context_frames]
        if self.transform_gt:
            ground_truth = self.transform_gt(ground_truth)

        # Stack grayscale frames as channels -> shape (context_length, H, W)
        context_frames = torch.cat(context_frames, dim=0)
        # Squeeze ground_truth to (H, W)
        ground_truth = ground_truth.squeeze(0)

        # Convert actions list to tensor
        context_actions = torch.tensor(context_actions)

        # Return ((frames_as_channels, actions), target_frame)
        return (context_frames, context_actions), ground_truth


    def train_val_split(self, val_ratio: float = 0.2, random_seed: int = 42):
        """
        Splits the dataset into train and validation subsets.

        Args:
            val_ratio (float): Fraction of data for validation (default 0.2).
            random_seed (int): Seed for reproducibility (default 42).

        Returns:
            (train_dataset, val_dataset)
        """
        total_len = len(self)
        val_len = int(val_ratio * total_len)
        train_len = total_len - val_len
        train_ds, val_ds = random_split(
            self,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(random_seed)
        )
        return train_ds, val_ds

class BinarizeFrame:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        return (tensor > self.threshold).float()
