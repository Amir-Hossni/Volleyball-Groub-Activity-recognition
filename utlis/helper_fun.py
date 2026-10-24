from pathlib import Path
import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def wrap_dp(model: nn.Module, device: torch.device) -> nn.Module:
    """Move model to device, then wrap with DataParallel when > 1 GPU is available."""
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"[DP] Using DataParallel on {num_gpus} GPUs: {gpu_names}")
        model = torch.nn.DataParallel(model)
    return model


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    """Load model weights from a checkpoint file (best_*.pth format),
    handling DataParallel 'module.' prefix differences safely."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        # Fallback 1: check current working directory
        alt_path = Path(ckpt_path.name)
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            # Fallback 2: search in /kaggle/input/ for checkpoints attached from previous notebook runs
            kaggle_input = Path("/kaggle/input")
            matches = list(kaggle_input.glob(f"**/{ckpt_path.name}")) if kaggle_input.exists() else []
            if matches:
                ckpt_path = matches[0]
                print(f"[ckpt] Auto-located checkpoint in /kaggle/input: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"[ckpt] Checkpoint file not found at '{path}' (nor in current directory, nor in /kaggle/input). "
                    f"Please ensure earlier stage checkpoints exist before running dependent models."
                )

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Determine whether target model and source checkpoint have 'module.' prefix
    target_keys = list(model.state_dict().keys())
    target_has_prefix = any(k.startswith("module.") for k in target_keys)
    src_has_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if target_has_prefix and not src_has_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not target_has_prefix and src_has_prefix:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"[ckpt] Loaded weights from {ckpt_path}")


def make_adamw(params, cfg: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )


def make_reduce_lr(optimizer, cfg: dict):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg["mode"],
        factor=cfg["factor"],
        patience=cfg["patience"],
        min_lr=cfg["min_lr"],
    )


def make_cosine_lr(optimizer, cfg: dict):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["T_max"],
        eta_min=cfg["eta_min"],
    )
