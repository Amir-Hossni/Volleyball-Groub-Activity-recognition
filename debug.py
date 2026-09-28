
from pathlib import Path
import copy
import yaml
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model


from engine.adapters import flatten_person_batch, identity_adapter
from Models.Baseline3.model_B3 import PersonClassifierB3
from Models.Baseline6.model_B6 import B6GroupActivityClassifier


from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
config = load_config("config.yaml")


# Load Config
# ==========================

data_cfg = config["Data"]
data_root = Path(data_cfg["DATA_ROOT"])
videos_path = data_root / data_cfg["PATHS"]["VIDEOS_PATH"]
annot_root = data_root / data_cfg["PATHS"]["TRACKING_ANNOTATION_PATH"]
pkl_path = Path(data_cfg["PATHS"]["PKL_PATH"])
scene_to_idx = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]
train_ids = data_cfg["SPLIT"]["TRAIN_IDS"]
val_ids = data_cfg["SPLIT"]["VAL_IDS"]


# Transform
transform = prepare_model(image_level=False)



##datset_new_ver
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

# # DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# B3
#stage1
person_model = PersonClassifierB3(
    num_classes=len(player_to_idx),
    pretrained=False
)

#stage2
checkpoint = torch.load(
    "/kaggle/working/best_B3_person_stage1.pth",
    map_location=device
)

person_model.load_state_dict(
    checkpoint["model_state_dict"]
)
# extract backbone
backboneB3 = person_model.model

#Basline6
backboneB6 = copy.deepcopy(backboneB3)

model_B6 = B6GroupActivityClassifier(backbone=backboneB6)


model = model_B6
# if torch.cuda.device_count() > 1:
#     print("Using DataParallel")
#     model = torch.nn.DataParallel(model)

model = model.to(device)


# Loss
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=-1
)



# Optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.1,
#     patience=2,
#     min_lr=1e-6
# )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)

# ============================================================
# Simple Training Loop
# ============================================================



num_epochs = 50
best_val_acc = 0.0

save_path = "/kaggle/working/best_B6_simple.pth"


for epoch in range(num_epochs):

    # ========================================================
    # Train
    # ========================================================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        images = batch["images"].to(device, non_blocking=True)
        targets = batch["scene_label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images, mask)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == targets).sum().item()
        train_total += targets.size(0)

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * train_correct / train_total


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["images"].to(device, non_blocking=True)
            targets = batch["scene_label"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            outputs = model(images, mask)

            loss = criterion(outputs, targets)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == targets).sum().item()
            val_total += targets.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * val_correct / val_total

    val_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler.step()


    # ========================================================
    # Logging
    # ========================================================

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Loss: {val_loss:.4f} "
        f"Validation Accuracy: {val_accuracy:.2f}% "
        f"Validation F1: {val_f1:.4f}"
    )

    print("=========================")


    # ========================================================
    # Save Best Model
    # ========================================================

    if val_accuracy > best_val_acc:

        best_val_acc = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": (
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_accuracy,
                "val_f1": val_f1,
            },
            save_path
        )

        print(
            f"Best model saved "
            f"(Val Accuracy = {val_accuracy:.2f}%)"
        )

        # Confusion Matrix
        cm = confusion_matrix(
            all_targets,
            all_predictions,
            labels=list(range(len(scene_to_idx)))
        )

        plt.figure(figsize=(8, 7))
        plt.imshow(cm)
        plt.colorbar()

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Validation Confusion Matrix")

        plt.xticks(
            range(len(scene_to_idx)),
            list(scene_to_idx.keys()),
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(scene_to_idx)),
            list(scene_to_idx.keys())
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        cm_path = "/kaggle/working/confusion_matrix_val.png"
        plt.savefig(cm_path)
        plt.close()

        print(
            f"Confusion matrix saved to {cm_path}"
        )