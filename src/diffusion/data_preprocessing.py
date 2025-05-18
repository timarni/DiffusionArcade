import pandas as pd

from datasets import Dataset
from tqdm import tqdm

def create_context_dataset(dataset, context_length, step=2):
    """
    Transforms the dataset into a dictionary with the following structure:

    {
        "context_frames": context_length number of frames before the ground truth frame, spaced by 'step',
        "context_actions": context_length number of actions before the ground truth frame, corresponding to the context frames,
        "ground_truth": The frame after the frames in context_frames, spaced by 'step'
    }

    For example, with context_length=4 and step=2, the function will use frames 1, 3, 5, 7 to predict frame 9.

    Inputs
    --------
    dataset : Dataset
        The dataset that you want to use. It should have the features: ['image', 'label', 'action', 'episode', 'step']
    context_length : int
        The number of desired context frames for the dataset
    step : int
        The step size between context frames (default 2 for every other frame)

    Returns
    --------
    Dataset
        dataset with the above structure
 
    """
    new_data = []

    grouped = dataset.to_pandas().groupby("episode")  # Only for sorting/grouping

    for episode, group in tqdm(grouped, desc="Processing dataset"):
        group = group.sort_values("step").reset_index()
        max_i = len(group)
        min_i = context_length * step
        for i in range(min_i, max_i, step):
            idxs = [group.loc[i - step * (context_length - j), "index"] for j in range(context_length)]
            gt_idx = int(group.loc[i, "index"])

            # Avoid losing image info by referencing dataset by index
            context_frames = [dataset[int(idx)]["image"] for idx in idxs]
            context_actions = [dataset[int(idx)]["action"] for idx in idxs]
            ground_truth = dataset[gt_idx]["image"]

            new_data.append({
                "context_frames": context_frames,
                "context_actions": context_actions,
                "ground_truth": ground_truth,
            })

    return Dataset.from_list(new_data)

