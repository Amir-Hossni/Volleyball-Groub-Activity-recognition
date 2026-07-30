import torch



def identity_adapter(batch, input_key="images", target_key="scene_label"):
    """
    Default adapter.

    Used for normal cases:
    image -> CNN
    frames -> Temporal model
    """

    inputs = batch[input_key]
    targets = batch[target_key]

    return inputs, targets





# def flatten_person_batch(
#     batch,
#     input_key="images",
#     target_key="player_labels",
#     ignore_index=-1
# ):
#     """
#     Convert:
#     images:
#         (B,P,C,H,W)

#     labels:
#         (B,P)

#     into:

#     images:
#         (N,C,H,W)

#     labels:
#         (N,)

#     removing padded players.
#     """


#     images = batch[input_key]

#     labels = batch[target_key]


#     B, P, C, H, W = images.shape


#     images = images.reshape(
#         B * P,
#         C,
#         H,
#         W
#     )


#     labels = labels.reshape(-1)


#     if ignore_index is not None:

#         mask = labels != ignore_index

#         images = images[mask]

#         labels = labels[mask]


#     return images, labels