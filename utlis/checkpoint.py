import torch
from pathlib import Path



def save_checkpoint(path,model,optimizer,epoch,metrics):

    path = Path(path)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            **metrics
        },
        path
    )



def load_checkpoint(path,model,optimizer=None,device="cpu"):

    checkpoint = torch.load(path,map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )


    return checkpoint["epoch"], checkpoint