

from pathlib import Path


import yaml
from torch.utils.data import DataLoader
import torch

from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model

from engine.trainer import Trainer
from engine.adapters import flatten_person_batch

# from Baseline2.model_B2 import B2Model
from Baseline3.model_B3 import PersonClassifierB3, GroupClassifierB3

# from Baseline2.training_B2 import train


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
    mode="person_grouped",
    transform=transform
)

val_dataset = VolleyballDataset(
    videos_path=videos_path,
    pkl_path=pkl_path,
    split_ids=val_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person_grouped",
    transform=transform
)


# # DataLoader
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
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=-1
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


trainer = Trainer(

    model=model,

    optimizer=optimizer,

    criterion=criterion,

    device=device,

    adapter=lambda batch:
        flatten_person_batch(
            batch,
            input_key="images",
            target_key="player_labels",
            ignore_index=-1
        ),

    num_classes=len(player_to_idx),

    save_path=
    "/kaggle/working/best_B3_person_model.pth",

    log_name="B3_person",

    epochs=50

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
    
    
    # trainer.fit(

    #     train_loader,

    #     val_loader

    # )
    
    # create_pkl_version(videos_root=videos_path,annot_root=annot_root,save_path= "/kaggle/working/annot_all.pkl")
    
    # batch = next(iter(train_loader))

    # images = batch["images"]
    # labels = batch["player_labels"]

    # print(images.shape)
    # print(labels.shape)

    # print("Valid players:")
    # print((labels != -1).sum())

    # print("=" * 50)
    # print(f"Train Dataset      : {len(train_dataset):,}")
    # print(f"Validation Dataset : {len(val_dataset):,}")
    # print(f"Train Loader       : {len(train_loader):,}")
    # print(f"Validation Loader  : {len(val_loader):,}")
    # print("=" * 50)


    # players = []

    # for i in range(50):

    #     sample = train_dataset[i]

    #     valid = (sample["player_labels"] != -1).sum().item()

    #     players.append(valid)


    # print("Average Players :", sum(players)/len(players))
    # print("Min Players :", min(players))
    # print("Max Players :", max(players))
    
    import time

    print("Testing DataLoader only")

    start = time.perf_counter()

    for i, batch in enumerate(train_loader):

        if i == 100:
            break

    elapsed = time.perf_counter() - start

    print(f"100 batches time: {elapsed:.2f} sec")
    print(f"Average batch loading: {elapsed/100:.4f} sec")
    
    print(torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))