from pathlib import Path
import copy
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

from Models.Baseline1.model_B1 import SceneClassifierB1
from Models.Baseline2.model_B2 import B2Model
from Models.Baseline3.model_B3 import PersonClassifierB3, GroupClassifierB3
from Models.Baseline4.model_B4 import TemporalImageClassifierB4
from Models.Baseline5.model_B5 import GroupTemporalClassifierB5, PersonTemporalB5
from Models.Baseline6.model_B6 import B6GroupActivityClassifier
from Models.Baseline7.model_B7 import B7HierarchicalClassifier


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


# dataset
train_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=train_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="clip_frames_players",
    transform=transform,
)

val_dataset = VolleyballDataset(
    videos_path=videos_path,
    annot_root=annot_root,
    split_ids=val_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="clip_frames_players",
    transform=transform,
)

# sampler
# B3 stage A: no sampler (the scene-label sampler does not apply to player actions)
train_sampler = create_weighted_sampler(
    train_dataset,
    target_key="scene_label",
)

# DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# Model
################

# B1
model_B1 = SceneClassifierB1(
    num_classes=len(scene_to_idx),
    pretrained=True,
)

# B2
model_B2 = B2Model(
    num_players=12,
    num_classes=len(scene_to_idx),
    pretrained=True,
)

# B3
# stage 1
person_model = PersonClassifierB3(
    num_classes=len(player_to_idx),
    pretrained=False,
)

# stage 2
checkpoint = torch.load(
    "/kaggle/working/best_B3_person_stage1.pth",
    map_location=device,
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
    num_classes=8,
)

# Baseline4
checkpoint = torch.load(
    "/kaggle/working/best_Baseline1.pth",
    map_location=device,
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

# Baseline5

# stage 1
backboneB5 = copy.deepcopy(backboneB3)
model_B5_stage1 = PersonTemporalB5(
    backbone=backboneB5,
    num_classes=9,
    lstm_hidden=512,
    lstm_layers=1,
    dropout=0.2,
)

checkpoint = torch.load(
    "/kaggle/working/best_Baseline5_stage1.pth",
    map_location=device,
)

model_B5_stage1.load_state_dict(
    checkpoint["model_state_dict"]
)

# stage 2
checkpoint = torch.load(
    "/kaggle/working/best_Baseline5_stage2.pth",
    map_location=device,
)

model_b5 = GroupTemporalClassifierB5(person_model=model_B5_stage1)

# fine-tune B5_stageB with the loaded weights
model_b5.load_state_dict(
    checkpoint["model_state_dict"]
)

# Baseline6
backboneB6 = copy.deepcopy(backboneB3)

model_B6 = B6GroupActivityClassifier(backbone=backboneB6)

# Baseline7
# Stage-1 person model (frozen B3 backbone + LSTM1), then group stage on top
backboneB7 = copy.deepcopy(backboneB3)

model_B5_for_B7 = PersonTemporalB5(
    backbone=backboneB7,
    num_classes=len(player_to_idx),
    lstm_hidden=512,
    lstm_layers=1,
    dropout=0.2,
)

checkpoint = torch.load(
    "/kaggle/working/best_Baseline5_stage1.pth",
    map_location=device,
)

model_B5_for_B7.load_state_dict(
    checkpoint["model_state_dict"]
)

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

model = model_B7

if torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = torch.nn.DataParallel(model)

model = model.to(device)


# B3 stage A retrain: all 9 frames per clip (mode="person")
# fresh ImageNet init, own optimizer/loss — B6 settings below stay untouched

b3_stageA_model = PersonClassifierB3(
    num_classes=len(player_to_idx),
    pretrained=True,
)

if torch.cuda.device_count() > 1:
    b3_stageA_model = torch.nn.DataParallel(b3_stageA_model)

b3_stageA_model = b3_stageA_model.to(device)

b3_stageA_criterion = nn.CrossEntropyLoss()

b3_stageA_optimizer = torch.optim.AdamW(
    b3_stageA_model.parameters(),
    lr=1e-4,
)


# Loss
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,
)

# Optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)
# 1e-4
# 3e-4
# 1e-3

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=4,
    min_lr=1e-6,
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=50,
#     eta_min=1e-6
# )

# ── B7 Phase 1: Linear Probe ──────────────────────────────────────────────────
# person_model is fully frozen.
# Only the group-level components are trainable:
#   group_norm, group_projection, lstm (LSTM2),
#   classifier, scene_skip_proj, scene_conv1, scene_conv2
# player_proj is part of the skip connection on the person side and stays
# frozen here; it only opens in Phase 2 together with LSTM1.
# ─────────────────────────────────────────────────────────────────────────────
optimizer_B7_probe = torch.optim.AdamW(
    [
        {"params": model_B7.group_norm.parameters()},
        {"params": model_B7.group_projection.parameters()},
        {"params": model_B7.lstm.parameters()},
        {"params": model_B7.classifier.parameters()},
        {"params": model_B7.scene_skip_proj.parameters()},
        {"params": model_B7.scene_conv1.parameters()},
        {"params": model_B7.scene_conv2.parameters()},
    ],
    lr=1e-3,
    weight_decay=1e-4,
)


# Trainer
############################
trainer_Baseline1 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=lambda batch: identity_adapter(
        batch,
        input_key="image",
        target_key="scene_label",
    ),
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_Baseline1.pth",
    class_names=list(scene_to_idx),
    log_name="Baseline1",
    epochs=50,
    use_amp=True,
    grad_clip=None,
)

# trainer_b3_stage1 = Trainer(
#     model=b3_stageA_model,
#     optimizer=b3_stageA_optimizer,
#     criterion=b3_stageA_criterion,
#     device=device,
#     adapter=flatten_person_batch,
#     num_classes=len(player_to_idx),
#     save_path="/kaggle/working/best_B3_person_stage1_v2.pth",
#     class_names=list(player_to_idx),
#     log_name="B3_person_stage1_v2",
#     epochs=5,
#     use_amp=True,
#     grad_clip=None,
# )

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
        target_key="scene_label",
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
        ignore_index=-1,
    ),
    num_classes=len(player_to_idx),
    save_path="/kaggle/working/best_Baseline5_stage1.pth",
    class_names=list(player_to_idx),
    log_name="Baseline5_stage1",
    epochs=50,
    use_amp=True,
    grad_clip=1.0,
    scheduler=scheduler,
)

trainer_Baseline5_S2 = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=lambda batch: identity_adapter(
        batch,
        input_key="images",
        target_key="scene_label",
        mask="player_mask",
    ),
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_Baseline5_stage2.pth",
    class_names=list(scene_to_idx),
    log_name="Baseline5_stage2",
    epochs=50,
    use_amp=True,
    grad_clip=1.0,
    scheduler=scheduler,
)

trainer_Baseline6 = Trainer(
    model=model_B6,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    adapter=lambda batch: identity_adapter(
        batch,
        input_key="images",
        target_key="scene_label",
        mask="mask",
    ),
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_Baseline6.pth",
    class_names=list(scene_to_idx),
    log_name="Baseline6",
    epochs=50,
    use_amp=True,
    grad_clip=None,
    scheduler=scheduler,
)

# ── B7 Phase 1 Trainer ────────────────────────────────────────────────────────
# Runs for 10 epochs with the group head only.
# early_stop_patience=10 so it exits if val_acc stops improving.
# save_path writes the best probe checkpoint; Phase 2 will overwrite it
# only when it finds a better val_acc, so the best weights always survive.
# ─────────────────────────────────────────────────────────────────────────────
trainer_B7_probe = Trainer(
    model=model_B7,
    optimizer=optimizer_B7_probe,
    criterion=criterion,
    device=device,
    adapter=lambda batch: identity_adapter(
        batch,
        input_key="images",
        target_key="scene_label",
        mask="mask",
    ),
    num_classes=len(scene_to_idx),
    save_path="/kaggle/working/best_Baseline7.pth",
    class_names=list(scene_to_idx),
    log_name="Baseline7_probe",
    epochs=10,
    early_stop_patience=10,
    use_amp=True,
    grad_clip=1.0,
    scheduler=None,
)


if __name__ == "__main__":

    # ── Phase 1: Linear Probe ─────────────────────────────────────────────────
    # person_model (backbone + LSTM1) is fully frozen.
    # Only the group-level components train.
    # ─────────────────────────────────────────────────────────────────────────
    print("=== B7 Phase 1: Linear Probe ===")
    trainer_B7_probe.fit(train_loader, val_loader)

    # ── Unfreeze LSTM1 between phases ─────────────────────────────────────────
    # unfreeze_person_lstm() opens only LSTM1 weights.
    # The ResNet backbone and the 9-class action head stay frozen.
    # Must be called on the unwrapped model when DataParallel is active.
    # ─────────────────────────────────────────────────────────────────────────
    actual_model = model_B7.module if hasattr(model_B7, "module") else model_B7
    lstm1_params = actual_model.unfreeze_person_lstm()
    print("LSTM1 unfrozen for Phase 2")

    # ── B7 Phase 2: Joint Fine-tune ───────────────────────────────────────────
    # LSTM1 + player_proj train at a low lr (1e-4) to preserve the
    # person-level representations learned in Stage A.
    # All group-level components train at a higher lr (1e-3).
    # CosineAnnealingLR decays both groups together over 50 epochs.
    # ─────────────────────────────────────────────────────────────────────────
    optimizer_B7_finetune = torch.optim.AdamW(
        [
            # LSTM1: low lr to preserve Stage-A person representations
            {"params": lstm1_params,
             "lr": 1e-4, "weight_decay": 1e-4},

            # player_proj (skip connection): low lr, same as LSTM1
            {"params": actual_model.player_proj.parameters(),
             "lr": 1e-4, "weight_decay": 1e-4},

            # group-level components: higher lr
            {"params": actual_model.group_norm.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": actual_model.group_projection.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": actual_model.lstm.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": actual_model.classifier.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": actual_model.scene_skip_proj.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": actual_model.scene_conv1.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
            {"params": actual_model.scene_conv2.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4},
        ],
    )

    scheduler_B7_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_B7_finetune,
        T_max=50,
        eta_min=1e-6,
    )

    # ── Phase 2 Trainer ───────────────────────────────────────────────────────
    # Built here so it picks up the freshly constructed optimizer and
    # scheduler. Shares the same model object as Phase 1, so Phase 2
    # starts from the weights Phase 1 left behind.
    # early_stop_patience=15 gives more room because the joint fine-tune
    # can plateau briefly after the LR drop before recovering.
    # ─────────────────────────────────────────────────────────────────────────
    trainer_B7_finetune = Trainer(
        model=model_B7,
        optimizer=optimizer_B7_finetune,
        criterion=criterion,
        device=device,
        adapter=lambda batch: identity_adapter(
            batch,
            input_key="images",
            target_key="scene_label",
            mask="mask",
        ),
        num_classes=len(scene_to_idx),
        save_path="/kaggle/working/best_Baseline7.pth",
        class_names=list(scene_to_idx),
        log_name="Baseline7_finetune",
        epochs=50,
        early_stop_patience=15,
        use_amp=True,
        grad_clip=1.0,
        scheduler=scheduler_B7_finetune,
    )

    print("=== B7 Phase 2: Joint Fine-tune ===")
    trainer_B7_finetune.fit(train_loader, val_loader)