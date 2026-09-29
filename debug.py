
from pathlib import Path
from collections import Counter
import copy
import hashlib
import os
import sys
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
b3_checkpoint_path = "/kaggle/working/best_B3_person_stage1.pth"

checkpoint = torch.load(
    b3_checkpoint_path,
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
# B6 Feature Probe (diagnostic only — no training)
#
#   python debug.py --probe
#
# Question: do the frozen B3 features that B6 consumes carry
# group-activity information (-> investigate B6 head/training),
# or not (-> frozen B3 representation may be the bottleneck)?
#
#   B6 backbone (frozen B3)            same object B6 uses
#       -> player features (12, 2048)  target frame only
#       -> masked MAX over players     same logic as B6 forward
#       -> Logistic Regression         train split only
#
# No LSTM, no group_projection, no B6 classifier, no sampler,
# no label smoothing. Nothing in the pipeline is modified.
# ============================================================

PROBE_TARGET_INDEX = 5      # index of the annotated frame inside the 9-frame window
PROBE_CACHE_DIR = Path("/kaggle/working")
PROBE_SEED = 0


def _probe_class_names(scene_to_idx):
    # One name per class id, first key in config order (aliases skipped)
    names = {}
    for name, idx in scene_to_idx.items():
        names.setdefault(idx, name)
    return [names[i] for i in range(len(names))]


def _probe_check_target_frame(dataset, target_index):
    """Same frame order as VolleyballDataset.__getitem__ (sorted frame ids)."""
    mismatches = 0
    for sample in dataset.samples:
        frame_ids = sorted(sample["frame_boxes"].keys())
        if len(frame_ids) <= target_index or frame_ids[target_index] != int(sample["clip_id"]):
            mismatches += 1

    example = dataset.samples[0]
    example_frames = sorted(example["frame_boxes"].keys())
    return mismatches, example, example_frames


def _probe_checkpoint_sanity(backbone, checkpoint, checkpoint_path, model_b6):
    print("=== B3 CHECKPOINT / BACKBONE ===")
    print(f"Checkpoint path: {checkpoint_path}")
    stat = os.stat(checkpoint_path)
    print(f"Checkpoint size: {stat.st_size / 1e6:.1f} MB")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(
        f"Checkpoint epoch (0-based): {checkpoint.get('epoch')} | "
        f"val_acc: {checkpoint.get('val_acc')} | val_f1: {checkpoint.get('val_f1')}"
    )

    # Every backbone tensor must equal the checkpoint tensor (PersonClassifierB3 stores it under "model.")
    saved = checkpoint["model_state_dict"]
    compared, mismatched = 0, []
    for name, tensor in backbone.state_dict().items():
        key = f"model.{name}"
        if key not in saved:
            continue
        compared += 1
        if not torch.equal(tensor.detach().cpu(), saved[key].detach().cpu()):
            mismatched.append(name)
    print(f"Backbone tensors compared with checkpoint: {compared} | mismatched: {len(mismatched)} {mismatched[:5]}")

    # Freshly initialised BatchNorm has running_mean == 0 and running_var == 1
    bn = backbone.bn1
    print(
        f"bn1 running_mean |mean|: {bn.running_mean.abs().mean().item():.4f} | "
        f"running_var mean: {bn.running_var.mean().item():.4f} (random init would be 0 / 1)"
    )

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Backbone parameters: total {total:,} | trainable {trainable:,}")
    print(f"backbone.fc: {backbone.fc}")
    print(f"B6 model.training: {model_b6.training} | backbone.training: {backbone.training}")
    print()


def _probe_cache_metadata(split_name, dataset, checkpoint_path, checkpoint, target_index, use_amp):
    sample_keys = "|".join(f"{s['video_id']}/{s['clip_id']}" for s in dataset.samples)
    stat = os.stat(checkpoint_path)
    return {
        "probe_version": 1,
        "split": split_name,
        "split_ids": sorted(dataset.split_ids, key=int),
        "num_samples": len(dataset),
        "samples_md5": hashlib.md5(sample_keys.encode()).hexdigest(),
        "mode": dataset.mode,
        "transform": repr(dataset.transform),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime": int(stat.st_mtime),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "target_index": target_index,
        "amp": use_amp,
    }


@torch.no_grad()
def _probe_extract(dataset, backbone, device, target_index, use_amp, batch_size=8, num_workers=4):
    """
    Frozen B3 features of the valid players at the target frame.
    Returns masked-max and masked-mean pooled features (N, 2048), labels (N,)
    and raw statistics of the per-player features.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    backbone.eval()

    max_features, mean_features, labels = [], [], []
    stats = {
        "count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf"),
        "zeros": 0, "nan": 0, "inf": 0, "l2_sum": 0.0, "players": 0,
    }

    for step, batch in enumerate(loader):
        # Dataset output is unchanged: (B,9,12,3,224,224). Only the target frame is used.
        images = batch["images"][:, target_index]    # (B,12,3,224,224)
        mask = batch["mask"][:, target_index]        # (B,12)

        B, P = mask.shape
        flat_images = images.reshape(B * P, *images.shape[2:])
        flat_mask = mask.reshape(B * P)

        # Same as B6 forward: only real crops go through the backbone (autocast as in training)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            valid = backbone(flat_images[flat_mask].to(device, non_blocking=True))
        valid = valid.flatten(1).float()

        stats["count"] += valid.numel()
        stats["sum"] += valid.double().sum().item()
        stats["sum_sq"] += (valid.double() ** 2).sum().item()
        stats["min"] = min(stats["min"], valid.min().item())
        stats["max"] = max(stats["max"], valid.max().item())
        stats["zeros"] += (valid == 0).sum().item()
        stats["nan"] += torch.isnan(valid).sum().item()
        stats["inf"] += torch.isinf(valid).sum().item()
        stats["l2_sum"] += valid.norm(dim=1).sum().item()
        stats["players"] += valid.shape[0]

        # Restore player slots, missing players as zeros (as in B6)
        features = torch.zeros(B * P, valid.shape[1], device=device)
        features[flat_mask.to(device)] = valid
        features = features.reshape(B, P, -1)

        # Masked max pooling — same logic as B6 forward
        mask3 = mask.to(device).unsqueeze(-1)
        pooled_max = features.masked_fill(~mask3, float("-inf")).max(dim=1).values
        pooled_max = torch.where(torch.isinf(pooled_max), torch.zeros_like(pooled_max), pooled_max)

        # Masked mean pooling — secondary diagnostic only
        counts = mask.to(device).sum(dim=1, keepdim=True).clamp_min(1).float()
        pooled_mean = features.masked_fill(~mask3, 0.0).sum(dim=1) / counts

        max_features.append(pooled_max.cpu())
        mean_features.append(pooled_mean.cpu())
        labels.append(batch["scene_label"].cpu())

        if step % 50 == 0:
            print(f"  extracted {min((step + 1) * batch_size, len(dataset))}/{len(dataset)}")

    return torch.cat(max_features), torch.cat(mean_features), torch.cat(labels), stats


def _probe_load_or_extract(split_name, dataset, backbone, device, checkpoint_path, checkpoint, target_index, use_amp):
    cache_path = PROBE_CACHE_DIR / f"b6_probe_{split_name}.pt"
    metadata = _probe_cache_metadata(split_name, dataset, checkpoint_path, checkpoint, target_index, use_amp)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("metadata") == metadata:
            print(f"[{split_name}] cache HIT: {cache_path}")
            return cached
        differing = [
            k for k in metadata
            if cached.get("metadata", {}).get(k) != metadata[k]
        ]
        print(f"[{split_name}] cache STALE ({cache_path}) — differing keys: {differing} -> re-extracting")
    else:
        print(f"[{split_name}] no cache -> extracting ({len(dataset)} samples)")

    max_features, mean_features, labels, stats = _probe_extract(
        dataset, backbone, device, target_index, use_amp
    )
    result = {
        "metadata": metadata,
        "max": max_features,
        "mean": mean_features,
        "labels": labels,
        "player_stats": stats,
    }
    torch.save(result, cache_path)
    print(f"[{split_name}] cache saved: {cache_path}")
    return result


def _probe_feature_sanity(split_name, data):
    stats = data["player_stats"]
    mean = stats["sum"] / stats["count"]
    std = (stats["sum_sq"] / stats["count"] - mean ** 2) ** 0.5
    print(f"[{split_name}] per-player B3 features (valid crops only):")
    print(
        f"  players: {stats['players']} | dim: {stats['count'] // max(stats['players'], 1)} | "
        f"mean {mean:.4f} | std {std:.4f} | min {stats['min']:.4f} | max {stats['max']:.4f}"
    )
    print(
        f"  zeros: {100 * stats['zeros'] / stats['count']:.2f}% | NaN: {stats['nan']} | Inf: {stats['inf']} | "
        f"mean L2 norm: {stats['l2_sum'] / max(stats['players'], 1):.3f}"
    )

    for pooling in ("max", "mean"):
        x = data[pooling]
        print(f"[{split_name}] {pooling}-pooled features: samples {x.shape[0]} | dim {x.shape[1]}")
        print(
            f"  mean {x.mean().item():.4f} | std {x.std().item():.4f} | min {x.min().item():.4f} | "
            f"max {x.max().item():.4f} | zeros {100 * (x == 0).float().mean().item():.2f}% | "
            f"NaN {torch.isnan(x).sum().item()} | Inf {torch.isinf(x).sum().item()} | "
            f"mean L2 norm {x.norm(dim=1).mean().item():.3f}"
        )


def _probe_fit(x_train, y_train, x_val):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, random_state=PROBE_SEED),
    )
    probe.fit(x_train, y_train)
    return probe, probe.predict(x_train), probe.predict(x_val)


def _probe_side_diagnostic(class_names, y_true, y_pred):
    """
    Side comes from the group label itself (l_* / r_*), not from player coordinates.
    Only reported if every class name has an explicit l/r prefix and each action has both sides.
    """
    parsed = []
    for name in class_names:
        if len(name) < 3 or name[0] not in "lr" or name[1] not in "_-":
            return None
        parsed.append((name[0], name[2:].replace("-", "_")))

    actions = sorted({action for _, action in parsed})
    if any({side for side, action in parsed if action == a} != {"l", "r"} for a in actions):
        return None

    side_of = np.array([side == "r" for side, _ in parsed])
    action_of = np.array([actions.index(action) for _, action in parsed])

    true_side, pred_side = side_of[y_true], side_of[y_pred]
    true_action, pred_action = action_of[y_true], action_of[y_pred]
    action_ok = true_action == pred_action

    return {
        "action_acc": 100 * action_ok.mean(),
        "side_acc": 100 * (true_side == pred_side).mean(),
        "side_acc_given_action": 100 * (true_side[action_ok] == pred_side[action_ok]).mean() if action_ok.any() else float("nan"),
        "actions": actions,
    }


def run_b6_feature_probe(
    train_dataset,
    val_dataset,
    model_b6,
    checkpoint,
    checkpoint_path,
    scene_to_idx,
    device,
    target_index=PROBE_TARGET_INDEX,
):
    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)

    backbone = model_b6.backbone            # exactly the frozen B3 backbone B6 uses
    backbone.eval()
    use_amp = device.type == "cuda"         # B6 Trainer runs the backbone under autocast
    class_names = _probe_class_names(scene_to_idx)
    num_classes = len(class_names)

    _probe_checkpoint_sanity(backbone, checkpoint, checkpoint_path, model_b6)

    print("=== TARGET FRAME ===")
    for name, dataset in (("train", train_dataset), ("val", val_dataset)):
        mismatches, example, example_frames = _probe_check_target_frame(dataset, target_index)
        print(
            f"[{name}] frame index {target_index} == clip id (annotated frame) for "
            f"{len(dataset) - mismatches}/{len(dataset)} samples"
        )
    print(
        f"Example: video {example['video_id']} clip {example['clip_id']} | window {example_frames} | "
        f"index {target_index} -> frame {example_frames[target_index]}"
    )
    print(f"Transform: {train_dataset.transform}")
    print(f"Autocast during extraction: {use_amp}")
    print()

    print("=== FEATURE EXTRACTION ===")
    train = _probe_load_or_extract("train", train_dataset, backbone, device, checkpoint_path, checkpoint, target_index, use_amp)
    val = _probe_load_or_extract("val", val_dataset, backbone, device, checkpoint_path, checkpoint, target_index, use_amp)
    print()

    print("=== FEATURE SANITY ===")
    _probe_feature_sanity("train", train)
    _probe_feature_sanity("val", val)
    print()

    y_train = train["labels"].numpy()
    y_val = val["labels"].numpy()
    labels = list(range(num_classes))

    results = {}
    for pooling in ("max", "mean"):
        _, pred_train, pred_val = _probe_fit(train[pooling].numpy(), y_train, val[pooling].numpy())
        results[pooling] = {
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_acc": 100 * accuracy_score(y_train, pred_train),
            "val_acc": 100 * accuracy_score(y_val, pred_val),
            "train_f1": f1_score(y_train, pred_train, labels=labels, average="macro", zero_division=0),
            "val_f1": f1_score(y_val, pred_val, labels=labels, average="macro", zero_division=0),
        }

    probe_max = results["max"]

    print("=== B6 FEATURE PROBE ===")
    print()
    print("Probe: StandardScaler (fit on train only) + multinomial LogisticRegression "
          f"(lbfgs, C=1.0, max_iter=5000, random_state={PROBE_SEED})")
    print(f"Pooling: masked MAX over players | frame index {target_index}")
    print()
    print("Feature shape:")
    print(f"Train: {tuple(train['max'].shape)}")
    print(f"Val:   {tuple(val['max'].shape)}")
    print()
    print("Classes:")
    for i, name in enumerate(class_names):
        print(f"{i} {name}")
    print()
    print("Class distribution:")
    train_counts, val_counts = Counter(y_train.tolist()), Counter(y_val.tolist())
    for i, name in enumerate(class_names):
        print(f"  {name:<12} train {train_counts[i]:>5} | val {val_counts[i]:>5}")
    print()
    print(f"Train Accuracy: {probe_max['train_acc']:.2f}%")
    print(f"Val Accuracy: {probe_max['val_acc']:.2f}%")
    print()
    print(f"Train Macro-F1: {probe_max['train_f1']:.4f}")
    print(f"Val Macro-F1: {probe_max['val_f1']:.4f}")
    print()
    print("Classification Report (val):")
    print(classification_report(y_val, probe_max["pred_val"], labels=labels, target_names=class_names, digits=4, zero_division=0))
    print("Confusion Matrix (val, rows = true, cols = predicted):")
    cm = confusion_matrix(y_val, probe_max["pred_val"], labels=labels)
    print(" " * 13 + " ".join(f"{i:>5}" for i in labels))
    for i, name in enumerate(class_names):
        print(f"{i} {name:<10} " + " ".join(f"{v:>5}" for v in cm[i]))
    print()

    print("=== SIDE DIAGNOSTIC (val) ===")
    side = _probe_side_diagnostic(class_names, y_val, probe_max["pred_val"])
    if side is None:
        print("Side accuracy: NOT AVAILABLE — no explicit side target")
    else:
        print("Side taken from the group label itself (l_* / r_*), not from player coordinates.")
        print(f"Actions: {side['actions']}")
        print(f"Action accuracy (side ignored): {side['action_acc']:.2f}%")
        print(f"Side accuracy (all samples): {side['side_acc']:.2f}%")
        print(f"Side accuracy when action is correct: {side['side_acc_given_action']:.2f}%")
    print()

    print("=== POOLING COMPARISON ===")
    print()
    for pooling in ("max", "mean"):
        r = results[pooling]
        print(f"{pooling.upper()}:")
        print(f"Train Accuracy: {r['train_acc']:.2f}%")
        print(f"Val Accuracy: {r['val_acc']:.2f}%")
        print(f"Val Macro-F1: {r['val_f1']:.4f}")
        print()

    print("=== DIAGNOSTIC SUMMARY ===")
    print()
    print(f"Probe Val Accuracy: {probe_max['val_acc']:.2f}%")
    print(f"Probe Val Macro-F1: {probe_max['val_f1']:.4f}")
    print()
    print("Interpretation:")
    acc = probe_max["val_acc"]
    if acc <= 50:
        print("  <= 50%: supports the hypothesis that the frozen B3 representation (after max pooling)")
        print("  may be the bottleneck — a linear read-out of a single frame is not better than B6.")
    elif acc < 55:
        print("  50-55%: between the 'bottleneck' and 'mixed' ranges — weak evidence either way.")
    elif acc <= 65:
        print("  55-65%: mixed evidence — the frozen features carry some information B6 may not be")
        print("  using, but the representation itself also looks limited.")
    elif acc < 70:
        print("  65-70%: between the 'mixed' and 'informative' ranges — leans towards the B6 head/training.")
    else:
        print("  >= 70%: suggests useful group-activity information exists in the frozen features;")
        print("  the B6 head/training deserves investigation rather than B3.")
    print("  This probe alone does not prove the cause: it reads one frame, uses a linear model,")
    print("  and says nothing about temporal information.")

    return results


if "--probe" in sys.argv:
    run_b6_feature_probe(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_b6=model_B6,
        checkpoint=checkpoint,
        checkpoint_path=b3_checkpoint_path,
        scene_to_idx=scene_to_idx,
        device=device,
    )
    sys.exit(0)

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