#Baseline7 training entry point.
# Follows the Baseline6 workflow in main.py:
#   deepcopy(B3 backbone) -> PersonTemporalB5 (load Stage-1 checkpoint)
#   -> B7HierarchicalClassifier -> shared Trainer with identity_adapter.
# Run: python train_B7.py
from pathlib import Path
import copy
import yaml
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model

from engine.trainer import Trainer
from engine.adapters import identity_adapter

from Models.Baseline3.model_B3 import PersonClassifierB3
from Models.Baseline5.model_B5 import PersonTemporalB5
from Models.Baseline7.model_B7 import B7HierarchicalClassifier


BASE_DIR = Path(__file__).resolve().parent

B7_STAGE1_CHECKPOINT = "/kaggle/working/best_Baseline5_stage1.pth"


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_B7(device, config=None):
    """Build B7 exactly as main.py does; raise if Stage-1 is missing."""
    config = config or load_config(str(BASE_DIR / "config.yaml"))
    data_cfg = config["Data"]
    scene_to_idx = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
    player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]

    person_model = PersonClassifierB3(
        num_classes=len(player_to_idx),
        pretrained=False
    )
    ckpt = torch.load(
        "/kaggle/working/best_B3_person_stage1.pth",
        map_location=device
    )
    person_model.load_state_dict(ckpt["model_state_dict"])
    backboneB3 = person_model.model
    backboneB3.fc = nn.Identity()

    backboneB7 = copy.deepcopy(backboneB3)
    model_B5_for_B7 = PersonTemporalB5(
        backbone=backboneB7,
        num_classes=len(player_to_idx),
        lstm_hidden=512,
        lstm_layers=1,
        dropout=0.2
    )
    ckpt_path = Path(B7_STAGE1_CHECKPOINT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"B7 Stage-1 checkpoint not found: {ckpt_path}. "
            "Train Baseline5 Stage-1 first; random init is not allowed."
        )
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    try:
        model_B5_for_B7.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Incompatible Stage-1 checkpoint: {ckpt_path} ({exc})"
        ) from exc

    model_B7 = B7HierarchicalClassifier(
        person_model=model_B5_for_B7,
        num_classes=len(scene_to_idx),
        group_feature_dim=3000,
        hidden_dim=500,
        feature_dropout=0.0,
        use_player_proj=True,
        pool="max_mean",
        scene_head="conv",
    )
    return model_B7, scene_to_idx, player_to_idx


def main():
    config = load_config("config.yaml")
    data_cfg = config["Data"]
    data_root = Path(data_cfg["DATA_ROOT"])
    videos_path = data_root / data_cfg["PATHS"]["VIDEOS_PATH"]
    annot_root = data_root / data_cfg["PATHS"]["TRACKING_ANNOTATION_PATH"]
    scene_to_idx = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
    player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]
    train_ids = data_cfg["SPLIT"]["TRAIN_IDS"]
    val_ids = data_cfg["SPLIT"]["VAL_IDS"]

    transform = prepare_model(image_level=False)

    train_dataset = VolleyballDataset(
        videos_path=videos_path,
        annot_root=annot_root,
        split_ids=train_ids,
        scene_to_idx=scene_to_idx,
        player_to_idx=player_to_idx,
        mode="clip_frames_players",
        transform=transform
    )
    val_dataset = VolleyballDataset(
        videos_path=videos_path,
        annot_root=annot_root,
        split_ids=val_ids,
        scene_to_idx=scene_to_idx,
        player_to_idx=player_to_idx,
        mode="clip_frames_players",
        transform=transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_B7, _, _ = build_B7(device, config)

    # B6 workflow: shared model var, optional DataParallel, then to(device).
    # DataParallel wraps the exact instance passed to Trainer/optimizer.
    # NOTE: optimizer is built AFTER wrapping so its params alias the
    # wrapped module's params (no stale unwrapped copy on multi-GPU).
    model = model_B7
    if torch.cuda.device_count() > 1:
        print("Using DataParallel")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=4,
        min_lr=1e-6
    )

    trainer_Baseline7 = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        adapter=lambda batch: identity_adapter(
            batch,
            input_key="images",
            target_key="scene_label",
            mask="mask"
        ),
        num_classes=len(scene_to_idx),
        save_path="/kaggle/working/best_Baseline7.pth",
        class_names=list(scene_to_idx),
        log_name="Baseline7",
        epochs=50,
        use_amp=True,
        grad_clip=1.0,
        scheduler=scheduler
    )
    trainer_Baseline7.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
