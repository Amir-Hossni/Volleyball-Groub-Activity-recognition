
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# from Data.datasetDIC import VolleyballDataset
from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model


from engine.trainer import Trainer
from engine.sampler import create_weighted_sampler
from engine.adapters import flatten_person_batch, identity_adapter

from Baseline1.model_B1 import SceneClassifierB1
from Baseline2.model_B2 import B2Model
from Baseline3.model_B3 import PersonClassifierB3, GroupClassifierB3
from Baseline4.model_B4 import TemporalImageClassifierB4
from Baseline5.model_B5 import GroupTemporalClassifierB5 , PersonTemporalB5 , GroupTemporalClassifierB5V2

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



##datset_new_ver
train_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=train_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person_temporal",
    transform=transform
)


val_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=val_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person_temporal",
    transform=transform
)

# #sampler
# train_sampler = create_weighted_sampler(
#     train_dataset,
#     target_key="scene_label"
# )


# # DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    dataset=val_dataset,
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
#b1
model_B1 = SceneClassifierB1(
    num_classes=len(scene_to_idx),
    pretrained=True
)



#B2
model_B2 = B2Model(
    num_players=12,
    num_classes=len(scene_to_idx),
    pretrained=True
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
# remove classification head
backboneB3.fc = nn.Identity()
# build group model
group_model = GroupClassifierB3(
    backbone=backboneB3,
    num_players=12,
    num_classes=8)



#Baseline4
checkpoint = torch.load(
    "/kaggle/working/best_Baseline1.pth",
    map_location=device
)

model_B1.load_state_dict(
    checkpoint["model_state_dict"]
)
# extract backbone
backbone1 = model_B1.model
model_B4 = TemporalImageClassifierB4(
    num_classes=8,
    backbone=backbone1,
)



#Basline5

# stage1
backboneB5 = backboneB3
model_B5_stage1 = PersonTemporalB5(
    backbone=backboneB5,
    num_classes=9,
    lstm_hidden=512,
    lstm_layers=1,
    dropout=0.2
)


# stage2
checkpoint = torch.load(
    "/kaggle/working/best_Baseline5_stage1.pth",
    map_location=device
)

model_B5_stage1.load_state_dict(
    checkpoint["model_state_dict"]
)

# backboneB5 = model_B5_stage1.backbone
# lstmB5 = model_B5_stage1.lstm

# model = GroupTemporalClassifierB5V2(
#     backbone=backboneB5,
#     lstm=lstmB5,
#     num_classes=8,
# )

model = GroupTemporalClassifierB5(person_model=model_B5_stage1)


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
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)



#Trainer
############################
trainer_Baseline1 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=lambda batch: identity_adapter(
        batch,
        input_key="image",
        target_key="scene_label"
    ),
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_Baseline1.pth",
    class_names=list(scene_to_idx),
    log_name="Baseline1",
    epochs=50,
    use_amp=True,
    grad_clip=None,
)


trainer_b3_stage1 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=flatten_person_batch,
    num_classes=len(player_to_idx),
    save_path="/kaggle/working/best_B3_person_stage1.pth",
    class_names=list(player_to_idx),
    log_name="B3_person_stage1",
    epochs=50,
    use_amp=True,
    grad_clip=None,
)

trainer_b3_stage2 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=identity_adapter,
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_B3_group_stage2.pth",
    class_names=list(scene_to_idx),
    log_name="B3_group_stage2",
    epochs=50,
    use_amp=True,
    grad_clip=None,
)


trainer_Baseline4 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=lambda batch: identity_adapter(
        batch,
        input_key="frames",
        target_key="scene_label"
    ),
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_Baseline4.pth",
    class_names=list(scene_to_idx),
    log_name="Baseline4",
    epochs=50,
    use_amp=True,
    grad_clip=None,
)

trainer_Baseline5_S1 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=lambda batch: flatten_person_batch(
        batch,
        input_key="images",
        target_key="player_labels",
        ignore_index=-1
    ),
    num_classes=len(player_to_idx),
    save_path="/kaggle/working/best_Baseline5_stage1.pth",
    class_names=list(player_to_idx),
    log_name="Baseline5_stage1",
    epochs=50,
    use_amp=True,
    grad_clip=1.0,
    scheduler=scheduler
)



trainer_Baseline5_S2 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=identity_adapter,
    num_classes=len(player_to_idx),
    save_path="/kaggle/working/best_Baseline5_stage2.pth",
    class_names=list(player_to_idx),
    log_name="Baseline5_stage2",
    epochs=50,
    use_amp=True,
    grad_clip=1.0,
    scheduler=scheduler
)
if __name__ == "__main__":
    
    trainer_Baseline5_S2.fit(train_loader, val_loader)
    
    
    # # ============================
    # # DEBUG: Check player tracks
    # # ============================

    # sample = train_dataset.samples[0]

    # player_tracks = sample["player_tracks"]

    # frame_id = sorted(player_tracks[0].keys())[0]

    # print("\n===== BOX DEBUG =====")
    # print(f"Frame: {frame_id}")

    # for player_id in range(12):
    #     box = player_tracks[player_id][frame_id]["box"]

    #     print(
    #         f"Player {player_id}: "
    #         f"ID={box.player_ID}, "
    #         f"box={box.box}"
    #     )

    # print("=====================\n")
    # create_pkl_version(videos_root=videos_path,annot_root=annot_root,save_path= "/kaggle/working/annot_all.pkl")
    
    