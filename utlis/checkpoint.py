import torch
from pathlib import Path



def save_checkpoint(path,model,optimizer,epoch,metrics,scaler=None):

    path = Path(path)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        **metrics
    }

    # Only include scaler state when AMP scaler exists (optional, backward compatible)
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)



def load_checkpoint(path,model,optimizer=None,scaler=None,device="cpu"):

    checkpoint = torch.load(path,map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # Optional scaler restore (backward compatible: old checkpoints lack this key)
    if scaler is not None and "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )


    return checkpoint["epoch"], checkpoint
