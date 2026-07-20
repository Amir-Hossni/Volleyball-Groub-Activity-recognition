
from Data.dataset import VolleyballDataset
from Data.extract_features import prepare_model
from pathlib import Path
import yaml

from Baseline2.model_B2 import B2Model
from Baseline2.training_B2 import train
from torch.utils.data import DataLoader

import torch


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
scene_to_idx = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]
train_ids = data_cfg["SPLIT"]["TRAIN_IDS"]
val_ids = data_cfg["SPLIT"]["VAL_IDS"]


# Transform
transform = prepare_model(image_level=False)


# Dataset
train_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=train_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="frame",
    transform=transform
)

val_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=val_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="frame",
    transform=transform
)


# DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# Model
model = B2Model(
    num_players=12,
    num_classes=len(scene_to_idx),
    pretrained=True
)

model = model.to(device)


# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)



if __name__ == "__main__":
    
    
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=50,
        save_path="/kaggle/working/best_B2_model.pth"
    )
  