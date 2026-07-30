import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import yaml
from pathlib import Path


import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision import models

from PIL import Image

from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)
from Data.boxinfo import BoxInfo
from Data.volleyball_annot_loader import load_tracking_annot
from Data.preprocessing import prepare_model
from Data.new_dataset import VolleyballDatasetDirect


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





device = "cuda" if torch.cuda.is_available() else "cpu"


        
dataset_root = "/kaggle/input/datasets/ahmedmohamed365/volleyball"


    
    
root = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos"

data = []

for video_folder_name in os.listdir(root):
    current_video_folder_path = os.path.join(root, video_folder_name)

    if not os.path.isdir(current_video_folder_path):
        continue

    annotations_path = os.path.join(current_video_folder_path, "annotations.txt")

    if not os.path.isfile(annotations_path):
        continue

    video_id = int(video_folder_name)

    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            image_filename = parts[0]                 # 13456.jpg
            group_label = parts[1]                    # r_spike

            clip_id = os.path.splitext(image_filename)[0]   # 13456

            image_path = os.path.join(
                current_video_folder_path,
                clip_id,
                image_filename
            )

            tracking_path = os.path.join(
                "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
                str(video_id),
                clip_id,
                f"{clip_id}.txt"
            )

            data.append({
                "video_id": video_id,
                "clip_id": clip_id,
                "image_path": image_path,
                "tracking_path": tracking_path,
                "group_label": group_label
            })
            


            
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_data = []
val_data = []
test_data = []

for sample in data:
    if sample["video_id"] in train_videos:
        train_data.append(sample)

    elif sample["video_id"] in val_videos:
        val_data.append(sample)

    else:
        test_data.append(sample)
label_map = {
    "l_pass": 0,
    "r_pass": 1,
    "l_spike": 2,
    "r_spike": 3,
    "l_set": 4,
    "r_set": 5,
    "l_winpoint": 6,
    "r_winpoint": 7,

    "l-pass": 0,
    "r-pass": 1,
    "l-spike": 2,
    "r-spike": 3,
    "l-set": 4,
    "r-set": 5,
    "l-winpoint": 6,
    "r-winpoint": 7
}
class_names = [
    "l_pass",
    "r_pass",
    "l_spike",
    "r_spike",
    "l_set",
    "r_set",
    "l_winpoint",
    "r_winpoint"
]
transform = prepare_model(image_level=False)

# class VolleyballDataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, idx):
    
#         sample = self.data[idx]
    
#         image = Image.open(sample["image_path"]).convert("RGB")
    
#         frame_boxes = load_tracking_annot(sample["tracking_path"])
    
#         frame_id = list(frame_boxes.keys())[0]
    
#         player_crops = []
    
#         for box_info in frame_boxes[frame_id][:12]:
#             x1, y1, x2, y2 = box_info.box
    
#             crop = image.crop((x1, y1, x2, y2))
    
#             if self.transform:
#                 crop = self.transform(crop)
    
#             player_crops.append(crop)
    
#         while len(player_crops) < 12:
#             player_crops.append(torch.zeros(3, 224, 224))
    
#         player_crops = torch.stack(player_crops)
    
#         label = torch.tensor(
#             label_map[sample["group_label"]],
#             dtype=torch.long
#         )
    
#         return player_crops, label
    
# dataset = VolleyballDataset(train_data, transform)
LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 40

# train_dataset = VolleyballDataset(train_data, transform)
# val_dataset = VolleyballDataset(val_data, transform)
# test_dataset = VolleyballDataset(test_data, transform)


train_dataset = VolleyballDatasetDirect(
    data=train_data,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    transform=transform
)


val_dataset = VolleyballDatasetDirect(
    data=val_data,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=4,
#     persistent_workers=True,
#     prefetch_factor=2
# )
class B3Model(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Linear(2048, 8)

    def forward(self, x):
        # x: [B,12,3,224,224]
        
        B, P, C, H, W = x.shape
        x = x.view(B*P, C, H, W)
        features = self.backbone(x)
        features = features.view(B, P, 2048)
        features = features.mean(dim=1)
        out = self.classifier(features)

        return out
model = B3Model()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)


checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6 
)
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []
total_f1_validation_plot = []
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    total_acc_train = total_loss_train = total_loss_val = total_acc_val = 0
    seen_train = seen_val = 0

    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False)
    # for inputs, labels in train_bar:
    #     inputs = inputs.to(device, non_blocking=True)
    #     labels = labels.to(device, non_blocking=True)
    for batch in train_bar:

        inputs = batch["images"].to(
            device,
            non_blocking=True
        )

        labels = batch["scene_label"].to(
            device,
            non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()
        optimizer.step()

        batch_correct = (torch.argmax(outputs, 1) == labels).sum().item()
        total_acc_train += batch_correct
        seen_train += labels.size(0)

        train_bar.set_postfix(
            loss=f"{train_loss.item():.4f}",
            acc=f"{total_acc_train / seen_train * 100:.2f}%",
        )
        
    true_labels = []
    predicted_labels = []
    model.eval()
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False)
    with torch.no_grad():
        # for inputs, labels in val_bar:
        #     inputs = inputs.to(device, non_blocking=True)
        #     labels = labels.to(device, non_blocking=True)
        for batch in val_bar:

            inputs = batch["images"].to(
            device,
            non_blocking=True
            )

            labels = batch["scene_label"].to(
                device,
                non_blocking=True
            )
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()

            batch_correct = (torch.argmax(outputs, 1) == labels).sum().item()
            total_acc_val += batch_correct
            seen_val += labels.size(0)

            val_bar.set_postfix(
                loss=f"{val_loss.item():.4f}",
                acc=f"{total_acc_val / seen_val * 100:.2f}%",
            )

    avg_train_loss = total_loss_train / len(train_loader)
    avg_val_loss = total_loss_val / len(val_loader)
    scheduler.step()

    # Save checkpoint every epoch
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }
    
    torch.save(
        checkpoint,
        f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
    )
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
    
        torch.save(
            checkpoint,
            f"{checkpoint_dir}/best_model.pth"
        )
    
        print(f"The Best model saved (Val Loss = {avg_val_loss:.4f})")
    

    total_loss_train_plot.append(round(avg_train_loss, 4))
    total_loss_validation_plot.append(round(avg_val_loss, 4))
    total_acc_train_plot.append(round(total_acc_train / seen_train * 100, 4))
    total_acc_validation_plot.append(round(total_acc_val / seen_val * 100, 4))
    val_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro"
    )
    total_f1_validation_plot.append(round(val_f1, 4))
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Train Accuracy: {total_acc_train / seen_train * 100:.2f}% "
        f"Validation Loss: {avg_val_loss:.4f} "
        f"Validation Accuracy: {total_acc_val / seen_val * 100:.2f}%"
        f"Validation F1: {val_f1:.4f}"
    )
    print("=" * 25) 