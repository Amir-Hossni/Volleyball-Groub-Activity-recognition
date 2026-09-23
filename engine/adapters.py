import torch



def identity_adapter(batch, input_key="images", target_key="scene_label", mask=None):
    """
    Default adapter.

    Used for normal cases:
    image -> CNN
    frames -> Temporal model
    """

    inputs = batch[input_key]
    targets = batch[target_key]

    # B5
    if mask == "player_mask":
        player_mask = inputs.abs().sum(dim=(2, 3, 4, 5)) > 0
        return inputs, targets, player_mask
    #B6
    elif mask == "mask":    
        mask = inputs.abs().sum(dim=(3, 4, 5)) > 0
        return inputs, targets, mask

    return inputs, targets






def flatten_person_batch(
    batch,
    input_key="images",
    target_key="player_labels",
    ignore_index=-1,
):
    """
    Supports:

    B3:
        images: (B, P, C, H, W)
        labels: (B, P)

    B5:
        images: (B, P, T, C, H, W)
        labels: (B, P)

    Output:

    B3:
        images: (N, C, H, W)

    B5:
        images: (N, T, C, H, W)

    N = valid players
    """

    images = batch[input_key]
    labels = batch[target_key]

    # B5 / temporal
    if images.dim() == 6:

        B, P, T, C, H, W = images.shape

        images = images.reshape(B * P, T, C, H, W,)

    # B3 / single frame
    elif images.dim() == 5:

        B, P, C, H, W = images.shape

        images = images.reshape(B * P, C, H, W,)

    else:
        raise ValueError(
            f"Unexpected image shape: {images.shape}"
        )

    labels = labels.reshape(-1)

    # Remove padded players
    if ignore_index is not None:

        mask = labels != ignore_index

        images = images[mask]
        labels = labels[mask]

    return images, labels

