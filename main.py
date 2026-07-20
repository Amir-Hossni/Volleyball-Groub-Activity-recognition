from random import sample

from Data.dataset import VolleyballDataset
from Data.extract_features import prepare_model
from pathlib import Path
import yaml
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch


BASE_DIR = Path(__file__).resolve().parent

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
config = load_config("config.yaml")
data_root = Path(config["Data"]["DATA_ROOT"])



# ==========================
# Load Config
# ==========================

data_cfg = config["Data"]
data_root = Path(data_cfg["DATA_ROOT"])
videos_path = data_root / data_cfg["PATHS"]["VIDEOS_PATH"]
annot_root = data_root / data_cfg["PATHS"]["TRACKING_ANNOTATION_PATH"]
scene_to_idx = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]
player_to_idx = data_cfg["CATEGORIES"]["PLAYER_TO_IDX"]
train_ids = data_cfg["SPLIT"]["TRAIN_IDS"]


transform = prepare_model(image_level=False)


train_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=train_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person",
    transform=transform
)

sample = train_dataset[0]

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)



if __name__ == "__main__":
    
    
    
    
    print(type(sample))
    print(sample.keys())
    
    batch = next(iter(train_loader))


    print(type(batch))
    print(len(batch))

    for i, item in enumerate(batch):
        print(i, type(item))
    
    
    print("=" * 60)
    print("Batch Information")
    print("=" * 60)

    for key, value in batch.items():

        if hasattr(value, "shape"):
            print(f"{key:<15}: {value.shape}")
        else:
            print(f"{key:<15}: {type(value)}")
            
    batch = next(iter(train_loader))


    
    print("Image shape:", batch["image"].shape)

    print("Player label:")
    print(batch["player_label"][:10])

    print(type(batch["player_label"][0]))

    print("Scene label:")
    print(batch["scene_label"][:10])

    print(type(batch["scene_label"][0]))
    
    print("Unique player labels:")

    print(set(batch["player_label"]))
    
    print(batch["player_label"][:10])
    print(type(batch["player_label"][0]))
    