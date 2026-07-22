
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
import torch

from utlis.early_stopping import EarlyStopping
from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model

# from Baseline2.model_B2 import B2Model
from Baseline3.model_B3 import PersonClassifierB3, GroupClassifierB3

# from Baseline2.training_B2 import train
from Baseline3.training_person_B3 import train_person_B3

# from Data.create_annot_pkl import create_pkl_version



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


# Dataset
train_dataset = VolleyballDataset(
    videos_path=videos_path,
    pkl_path=pkl_path,
    split_ids=train_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person",
    transform=transform
)

val_dataset = VolleyballDataset(
    videos_path=videos_path,
    pkl_path=pkl_path,
    split_ids=val_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person",
    transform=transform
)


# DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
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


# Model
################

#B2
# model = B2Model(
#     num_players=12,
#     num_classes=len(scene_to_idx),
#     pretrained=True
# )

# B3
model = PersonClassifierB3(
    num_classes=len(player_to_idx),
    pretrained=True
)
# person_model.load_state_dict(torch.load("person.pth"))
# backbone = person_model.backbone
# group_model = GroupClassifierB3(backbone)

if torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = torch.nn.DataParallel(model)

model = model.to(device)


# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

early_stopping = EarlyStopping(
    patience=10,
    mode="max"
)

if __name__ == "__main__":
    
    
    # train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     device,
    #     epochs=50,
    #     save_path="/kaggle/working/best_B2_model.pth"
    # )
    
    
#     train_person_B3(
#     model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     device,
#     epochs=50,
#     save_path="/kaggle/working/best_B3_person_model.pth"
# )
    
    # create_pkl_version(videos_root=videos_path,annot_root=annot_root,save_path= "/kaggle/working/annot_all.pkl")
    import time

    print("Dataset size:", len(train_dataset))

    # Test Dataset directly
    print("\nTesting Dataset...")

    start = time.time()

    for i in range(10):

        sample = train_dataset[i]

        print(
            i,
            sample["image"].shape,
            sample["player_label"]
        )

    print("10 samples time:", time.time() - start)



    # Test DataLoader
    print("\nTesting DataLoader...")

    start = time.time()

    for batch_idx, batch in enumerate(train_loader):

        print(
            "batch:",
            batch_idx,
            batch["image"].shape
        )

        if batch_idx == 5:
            break


    print("6 batches time:", time.time() - start)