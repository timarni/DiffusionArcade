import pandas as pd
from datasets import Dataset

def create_context_dataset(dataset, context_length):
    """
    Transforms the dataset that we have into a dictionary with the following structure:

    {
        "context_frames": context_length number of frames before the ground truth frame
        "context_actions": context_length number of actions before the ground truth frame, corresponding to the context frames
        "ground_truth": The frame after the frames in context_frames
    }

    Inputs
    --------
    dataset : Dataset
        The dataset that you want to use. It should have the features: ['image', 'label', 'action', 'episode', 'step']
    context_length : int
        The number of desired context frames for the dataset

    Returns
    --------
    Dataset
        dataset with the above structure
 
    """
    new_data = []

    grouped = dataset.to_pandas().groupby("episode")  # Only for sorting/grouping

    for episode, group in grouped:
        group = group.sort_values("step").reset_index()

        for i in range(context_length, len(group)):
            idxs = group.loc[i - context_length:i - 1, "index"].tolist()
            gt_idx = int(group.loc[i, "index"])

            # Avoid losing image info by referencing dataset by index
            context_frames = [dataset[idx]["image"] for idx in idxs]
            context_actions = [dataset[idx]["action"] for idx in idxs]
            ground_truth = dataset[gt_idx]["image"]

            new_data.append({
                "context_frames": context_frames,
                "context_actions": context_actions,
                "ground_truth": ground_truth,
            })

    return Dataset.from_list(new_data)

