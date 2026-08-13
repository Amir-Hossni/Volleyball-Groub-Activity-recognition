import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(dataset, target_key="scene_label"):
    """
    Create a WeightedRandomSampler based on a target label.

    Parameters
    ----------
    dataset:
        VolleyballDataset instance.

    target_key:
        "scene_label" -> group activity sampling (8 classes)
        "player_label" -> person action sampling (9 classes)

    Returns
    -------
    WeightedRandomSampler
    """

    labels = []

    for sample in dataset.samples:

        if target_key == "scene_label":

            labels.append(sample["scene_label"])

        elif target_key == "player_label":

            # B3/B5 Stage 1 may store player labels differently
            if "player_label" in sample:
                labels.append(sample["player_label"])

            else:
                raise ValueError(
                    "Dataset samples do not contain 'player_label'."
                )

        else:
            raise ValueError(
                f"Unknown target_key: {target_key}"
            )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    class_counts = torch.bincount(labels)

    class_weights = torch.zeros_like(
        class_counts,
        dtype=torch.float
    )

    # Avoid division by zero for classes not present
    mask = class_counts > 0
    class_weights[mask] = 1.0 / class_counts[mask].float()

    sample_weights = class_weights[labels].double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler