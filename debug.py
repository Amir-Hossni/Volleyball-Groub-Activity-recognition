
# ============================================================
# VOLLEYBALL DATASET AUDIT
# ============================================================
# Run this AFTER creating:
#   train_dataset
#   val_dataset
#   transform
#
# The script audits:
# 1. Dataset sizes
# 2. Scene distribution
# 3. Player distribution
# 4. Number of players per sample
# 5. Frame consistency
# 6. Player-ID consistency
# 7. Player-label consistency
# 8. Tracking trimming [5:-6]
# 9. Missing players / padding
# 10. Input mask
# 11. Bounding-box visualization
# 12. Train/Val split statistics
# ============================================================

from pathlib import Path
from collections import Counter, defaultdict
import random
import math
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader

from Data.volleyball_annot_loader import load_tracking_annot
from Data.boxinfo import BoxInfo
from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model

# ============================================================
# CONFIG
# ============================================================

NUM_PLAYERS = 12
NUM_FRAMES = 9

NUM_CLIPS_TO_AUDIT = 10
NUM_VIS_TRAIN = 5
NUM_VIS_VAL = 5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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

# ============================================================
# 1. BASIC DATASET INFO
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC DATASET INFORMATION")
print("=" * 80)

print(f"Train samples : {len(train_dataset)}")
print(f"Val samples   : {len(val_dataset)}")

print("\nTrain videos:")
print(sorted(set(s["video_id"] for s in train_dataset.samples)))

print("\nVal videos:")
print(sorted(set(s["video_id"] for s in val_dataset.samples)))


# ============================================================
# 2. SCENE DISTRIBUTION
# ============================================================

def scene_distribution(dataset, name):
    counter = Counter()

    for sample in dataset.samples:
        counter[sample["scene_label"]] += 1

    print(f"\n{name} Scene Distribution")
    print("-" * 50)

    for label, count in sorted(counter.items()):
        print(f"class {label:<3} -> {count}")

    total = sum(counter.values())

    print(f"TOTAL = {total}")

    if total > 0:
        print("\nPercentages:")
        for label, count in sorted(counter.items()):
            print(
                f"class {label:<3} -> "
                f"{count / total * 100:.2f}%"
            )

    return counter


train_scene_counter = scene_distribution(
    train_dataset,
    "TRAIN"
)

val_scene_counter = scene_distribution(
    val_dataset,
    "VAL"
)


# ============================================================
# 3. SAMPLES PER VIDEO
# ============================================================

def samples_per_video(dataset, name):

    counter = Counter(
        sample["video_id"]
        for sample in dataset.samples
    )

    print(f"\n{name} Samples Per Video")
    print("-" * 50)

    for video_id, count in sorted(
        counter.items(),
        key=lambda x: int(x[0])
    ):
        print(f"video {video_id:<3} -> {count}")

    return counter


train_video_counter = samples_per_video(
    train_dataset,
    "TRAIN"
)

val_video_counter = samples_per_video(
    val_dataset,
    "VAL"
)


# ============================================================
# 4. GET PLAYER INFORMATION FROM DATASET SAMPLE
# ============================================================

def analyze_player_tracks(dataset, name):

    print("\n" + "=" * 80)
    print(f"4. PLAYER TRACK STATISTICS - {name}")
    print("=" * 80)

    player_counts = Counter()
    player_label_counter = Counter()

    missing_player_counter = Counter()

    for sample in dataset.samples:

        tracks = sample["player_tracks"]

        num_existing_players = 0

        for player_id in range(NUM_PLAYERS):

            track = tracks[player_id]

            if len(track) > 0:

                num_existing_players += 1

                # Check labels
                labels = [
                    item["label_idx"]
                    for item in track.values()
                ]

                player_label_counter.update(labels)

        missing_players = NUM_PLAYERS - num_existing_players

        player_counts[num_existing_players] += 1
        missing_player_counter[missing_players] += 1

    print("\nNumber of existing players per sample:")

    for num_players, count in sorted(player_counts.items()):
        print(
            f"{num_players:2d} players -> "
            f"{count} samples "
            f"({count / len(dataset) * 100:.2f}%)"
        )

    print("\nMissing players per sample:")

    for missing, count in sorted(missing_player_counter.items()):
        print(
            f"{missing:2d} missing -> "
            f"{count} samples "
            f"({count / len(dataset) * 100:.2f}%)"
        )

    print("\nPlayer action label distribution:")

    for label, count in sorted(player_label_counter.items()):
        print(f"class {label:<3} -> {count}")

    return player_counts


train_player_stats = analyze_player_tracks(
    train_dataset,
    "TRAIN"
)

val_player_stats = analyze_player_tracks(
    val_dataset,
    "VAL"
)


# ============================================================
# 5. DEEP TRACK AUDIT
# ============================================================

def deep_track_audit(dataset, name, num_samples=10):

    print("\n" + "=" * 80)
    print(f"5. DEEP TRACK AUDIT - {name}")
    print("=" * 80)

    if len(dataset.samples) == 0:
        print("Dataset is empty.")
        return

    indices = random.sample(
        range(len(dataset.samples)),
        min(num_samples, len(dataset.samples))
    )

    for sample_idx in indices:

        sample = dataset.samples[sample_idx]

        print("\n" + "-" * 80)

        print(
            f"sample_index={sample_idx} | "
            f"video={sample['video_id']} | "
            f"clip={sample['clip_id']} | "
            f"scene={sample['scene_label']}"
        )

        tracks = sample["player_tracks"]

        all_frame_ids = sorted({
            frame_id
            for player_id in range(NUM_PLAYERS)
            for frame_id in tracks[player_id]
        })

        print(
            f"Global frame IDs ({len(all_frame_ids)}): "
            f"{all_frame_ids}"
        )

        print("\nPer-player tracks:")

        for player_id in range(NUM_PLAYERS):

            track = tracks[player_id]

            if not track:
                print(
                    f"Player {player_id:2d}: MISSING"
                )
                continue

            frame_ids = sorted(track.keys())

            labels = [
                track[f]["label_idx"]
                for f in frame_ids
            ]

            unique_labels = sorted(set(labels))

            print(
                f"Player {player_id:2d}: "
                f"{len(frame_ids):2d} frames | "
                f"frames={frame_ids} | "
                f"labels={unique_labels}"
            )

            if len(unique_labels) > 1:
                print(
                    "   !!! WARNING: "
                    "PLAYER HAS MULTIPLE LABELS !!!"
                )


deep_track_audit(
    train_dataset,
    "TRAIN",
    NUM_CLIPS_TO_AUDIT
)

deep_track_audit(
    val_dataset,
    "VAL",
    NUM_CLIPS_TO_AUDIT
)


# ============================================================
# 6. CHECK WHETHER ALL PLAYERS SHARE THE SAME 9 FRAMES
# ============================================================

def frame_alignment_audit(dataset, name):

    print("\n" + "=" * 80)
    print(f"6. FRAME ALIGNMENT AUDIT - {name}")
    print("=" * 80)

    bad_samples = []

    for idx, sample in enumerate(dataset.samples):

        tracks = sample["player_tracks"]

        existing_tracks = [
            tracks[player_id]
            for player_id in range(NUM_PLAYERS)
            if tracks[player_id]
        ]

        if not existing_tracks:
            continue

        frame_sets = [
            set(track.keys())
            for track in existing_tracks
        ]

        reference = frame_sets[0]

        aligned = all(
            frames == reference
            for frames in frame_sets[1:]
        )

        if not aligned:
            bad_samples.append(idx)

    print(f"Total samples: {len(dataset)}")
    print(f"Misaligned samples: {len(bad_samples)}")

    if bad_samples:
        print(
            "\nFirst problematic sample indices:"
        )
        print(bad_samples[:30])

    return bad_samples


train_bad_alignment = frame_alignment_audit(
    train_dataset,
    "TRAIN"
)

val_bad_alignment = frame_alignment_audit(
    val_dataset,
    "VAL"
)


# ============================================================
# 7. CHECK PLAYER LABEL CONSISTENCY
# ============================================================

def player_label_consistency(dataset, name):

    print("\n" + "=" * 80)
    print(f"7. PLAYER LABEL CONSISTENCY - {name}")
    print("=" * 80)

    inconsistent = []

    for idx, sample in enumerate(dataset.samples):

        tracks = sample["player_tracks"]

        for player_id in range(NUM_PLAYERS):

            track = tracks[player_id]

            if not track:
                continue

            labels = {
                item["label_idx"]
                for item in track.values()
            }

            if len(labels) > 1:

                inconsistent.append(
                    (
                        idx,
                        sample["video_id"],
                        sample["clip_id"],
                        player_id,
                        sorted(labels)
                    )
                )

    print(
        f"Inconsistent player labels: "
        f"{len(inconsistent)}"
    )

    for item in inconsistent[:30]:
        print(item)

    return inconsistent


train_bad_labels = player_label_consistency(
    train_dataset,
    "TRAIN"
)

val_bad_labels = player_label_consistency(
    val_dataset,
    "VAL"
)


# ============================================================
# 8. DIRECT AUDIT OF ORIGINAL ANNOTATION TRIMMING
# ============================================================

def audit_trimming(
    dataset,
    name,
    num_samples=10
):

    print("\n" + "=" * 80)
    print(f"8. ORIGINAL ANNOTATION TRIMMING AUDIT - {name}")
    print("=" * 80)

    indices = random.sample(
        range(len(dataset.samples)),
        min(num_samples, len(dataset.samples))
    )

    for sample_idx in indices:

        sample = dataset.samples[sample_idx]

        video_id = sample["video_id"]
        clip_id = sample["clip_id"]

        tracking_file = (
            dataset.annot_root
            / video_id
            / clip_id
            / f"{clip_id}.txt"
        )

        print("\n" + "-" * 80)

        print(
            f"video={video_id}, "
            f"clip={clip_id}"
        )

        print(f"file={tracking_file}")

        if not tracking_file.exists():
            print("ANNOTATION FILE DOES NOT EXIST")
            continue

        player_boxes = {
            idx: []
            for idx in range(NUM_PLAYERS)
        }

        with open(tracking_file, "r") as file:

            for line in file:

                box_info = BoxInfo(line)

                if box_info.player_ID > 11:
                    continue

                player_boxes[
                    box_info.player_ID
                ].append(box_info)

        for player_id in range(NUM_PLAYERS):

            boxes = player_boxes[player_id]

            if not boxes:
                continue

            original_frames = [
                b.frame_ID
                for b in boxes
            ]

            trimmed = boxes[5:]
            trimmed = trimmed[:-6]

            trimmed_frames = [
                b.frame_ID
                for b in trimmed
            ]

            print(
                f"\nPlayer {player_id}:"
            )

            print(
                f"  original count = "
                f"{len(original_frames)}"
            )

            print(
                f"  original frames = "
                f"{original_frames}"
            )

            print(
                f"  trimmed count = "
                f"{len(trimmed_frames)}"
            )

            print(
                f"  trimmed frames = "
                f"{trimmed_frames}"
            )


audit_trimming(
    train_dataset,
    "TRAIN"
)

audit_trimming(
    val_dataset,
    "VAL"
)


# ============================================================
# 9. LOST / GENERATED / GROUPING AUDIT
# ============================================================

def annotation_flags_audit(dataset, name):

    print("\n" + "=" * 80)
    print(f"9. ANNOTATION FLAGS AUDIT - {name}")
    print("=" * 80)

    lost_counter = Counter()
    generated_counter = Counter()
    grouping_counter = Counter()

    total_boxes = 0

    for sample in dataset.samples:

        tracks = sample["player_tracks"]

        for player_id in range(NUM_PLAYERS):

            for item in tracks[player_id].values():

                box = item["box"]

                total_boxes += 1

                lost_counter[
                    getattr(box, "lost", None)
                ] += 1

                generated_counter[
                    getattr(box, "generated", None)
                ] += 1

                grouping_counter[
                    getattr(box, "grouping", None)
                ] += 1

    print(f"Total boxes: {total_boxes}")

    print("\nLOST:")
    print(lost_counter)

    print("\nGENERATED:")
    print(generated_counter)

    print("\nGROUPING:")
    print(grouping_counter)


annotation_flags_audit(
    train_dataset,
    "TRAIN"
)

annotation_flags_audit(
    val_dataset,
    "VAL"
)


# ============================================================
# 10. ACTUAL DATASET OUTPUT AUDIT
# ============================================================

def dataset_output_audit(dataset, name):

    print("\n" + "=" * 80)
    print(f"10. ACTUAL __getitem__ OUTPUT - {name}")
    print("=" * 80)

    idx = random.randrange(len(dataset))

    sample = dataset[idx]

    print(f"Sample index: {idx}")

    for key, value in sample.items():

        if torch.is_tensor(value):

            print(
                f"{key}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"min={value.min().item():.5f}, "
                f"max={value.max().item():.5f}"
            )

        else:

            print(
                f"{key}: "
                f"type={type(value)}, "
                f"value={value}"
            )


dataset_output_audit(
    train_dataset,
    "TRAIN"
)

dataset_output_audit(
    val_dataset,
    "VAL"
)


# ============================================================
# 11. CHECK ZERO-PADDING
# ============================================================

def padding_audit(dataset, name):

    print("\n" + "=" * 80)
    print(f"11. ZERO PADDING AUDIT - {name}")
    print("=" * 80)

    zero_players = Counter()

    nonzero_players = Counter()

    suspicious = []

    for idx in range(
        min(len(dataset), 500)
    ):

        sample = dataset[idx]

        images = sample["images"]

        # images:
        # (12, 9, C, H, W)

        player_energy = (
            images.abs()
            .sum(dim=(1, 2, 3, 4))
        )

        for player_id in range(NUM_PLAYERS):

            energy = player_energy[player_id].item()

            if energy == 0:
                zero_players[player_id] += 1
            else:
                nonzero_players[player_id] += 1

    print("\nZero players:")

    for player_id in range(NUM_PLAYERS):

        print(
            f"player {player_id:2d}: "
            f"zero={zero_players[player_id]:4d}, "
            f"nonzero={nonzero_players[player_id]:4d}"
        )


padding_audit(
    train_dataset,
    "TRAIN"
)


# ============================================================
# 12. PLAYER MASK AUDIT
# ============================================================

def player_mask_audit(dataset, name):

    print("\n" + "=" * 80)
    print(f"12. PLAYER MASK AUDIT - {name}")
    print("=" * 80)

    mismatch_count = 0
    checked = 0

    for idx in range(
        min(len(dataset), 500)
    ):

        sample = dataset[idx]

        images = sample["images"]

        mask_from_tensor = (
            images.abs()
            .sum(dim=(1, 2, 3, 4))
            > 0
        )

        tracks = sample["player_tracks"]

        mask_from_annotation = torch.tensor(
            [
                bool(tracks[player_id])
                for player_id in range(NUM_PLAYERS)
            ],
            dtype=torch.bool
        )

        if not torch.equal(
            mask_from_tensor,
            mask_from_annotation
        ):

            mismatch_count += 1

            if mismatch_count <= 20:

                print(
                    f"\nMismatch sample {idx}"
                )

                print(
                    "Tensor mask :",
                    mask_from_tensor.tolist()
                )

                print(
                    "Annot mask  :",
                    mask_from_annotation.tolist()
                )

        checked += 1

    print(
        f"\nChecked: {checked}"
    )

    print(
        f"Mask mismatches: "
        f"{mismatch_count}"
    )

    print(
        f"Mismatch percentage: "
        f"{mismatch_count / max(checked, 1) * 100:.2f}%"
    )


player_mask_audit(
    train_dataset,
    "TRAIN"
)

player_mask_audit(
    val_dataset,
    "VAL"
)


# ============================================================
# 13. DATALOADER BATCH AUDIT
# ============================================================

def dataloader_audit(dataset, name):

    print("\n" + "=" * 80)
    print(f"13. DATALOADER BATCH AUDIT - {name}")
    print("=" * 80)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    batch = next(iter(loader))

    print("\nBatch contents:")

    for key, value in batch.items():

        if torch.is_tensor(value):

            print(
                f"{key}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )

        else:

            print(
                f"{key}: "
                f"type={type(value)}"
            )

    images = batch["images"]

    print("\nExpected:")
    print(
        "(B, 12, 9, C, H, W)"
    )

    print(
        f"Actual: {tuple(images.shape)}"
    )


dataloader_audit(
    train_dataset,
    "TRAIN"
)

dataloader_audit(
    val_dataset,
    "VAL"
)


# ============================================================
# 14. VISUALIZE COMPLETE 9-FRAME TRACK
# ============================================================

def denormalize_if_needed(tensor):

    # ImageNet normalization used by most pretrained models.
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    x = tensor.cpu() * std + mean

    return x.clamp(0, 1)


def visualize_sample(
    dataset,
    sample_idx,
    save_dir="audit_visualizations"
):

    sample = dataset.samples[sample_idx]

    video_id = sample["video_id"]
    clip_id = sample["clip_id"]

    tracks = sample["player_tracks"]

    frame_ids = sorted({
        frame_id
        for player_id in range(NUM_PLAYERS)
        for frame_id in tracks[player_id]
    })[:NUM_FRAMES]

    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14)
    )

    axes = axes.flatten()

    for ax, frame_id in zip(
        axes,
        frame_ids
    ):

        image_path = (
            sample["clip_path"]
            / f"{frame_id}.jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)

        boxes = []

        for player_id in range(NUM_PLAYERS):

            track = tracks[player_id]

            if frame_id not in track:
                continue

            box = track[frame_id]["box"]

            x1, y1, x2, y2 = box.box

            category = box.category

            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                max(0, y1 - 5),
                f"P{player_id}:{category}",
                fontsize=7
            )

        ax.set_title(
            f"Frame {frame_id}"
        )

        ax.axis("off")

    for ax in axes[len(frame_ids):]:
        ax.axis("off")

    fig.suptitle(
        f"Video {video_id} | Clip {clip_id}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        save_dir
        / f"video_{video_id}_clip_{clip_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# TRAIN VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("14A. TRAIN VISUALIZATION")
print("=" * 80)

train_indices = random.sample(
    range(len(train_dataset.samples)),
    min(NUM_VIS_TRAIN, len(train_dataset.samples))
)

for idx in train_indices:

    visualize_sample(
        train_dataset,
        idx,
        save_dir="audit_visualizations/train"
    )


# ============================================================
# VAL VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("14B. VAL VISUALIZATION")
print("=" * 80)

val_indices = random.sample(
    range(len(val_dataset.samples)),
    min(NUM_VIS_VAL, len(val_dataset.samples))
)

for idx in val_indices:

    visualize_sample(
        val_dataset,
        idx,
        save_dir="audit_visualizations/val"
    )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("15. AUDIT FINISHED")
print("=" * 80)

print("""
Please send me:

1. COMPLETE CONSOLE OUTPUT

2. The generated folders:
   audit_visualizations/train
   audit_visualizations/val

3. Especially these results:
   - Frame alignment
   - Player label consistency
   - LOST / GENERATED / GROUPING
   - Padding audit
   - Player mask audit
   - Dataset shapes

Do NOT modify the dataset based on these results yet.
We will first determine whether there is actually a data problem.
""")
