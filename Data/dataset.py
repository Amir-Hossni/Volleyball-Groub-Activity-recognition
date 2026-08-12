from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .boxinfo import BoxInfo
from .volleyball_annot_loader import load_tracking_annot, load_video_annot


class VolleyballDataset(Dataset):

    def __init__(
        self,
        videos_path,
        annot_root,
        split_ids,
        scene_to_idx,
        player_to_idx=None,
        mode="clip_frames",
        transform=None,
    ):
        self.videos_path = Path(videos_path)
        self.annot_root = Path(annot_root)

        self.split_ids = (
            set(split_ids) if not isinstance(split_ids, set) else split_ids
        )

        self.scene_to_idx = scene_to_idx
        self.player_to_idx = player_to_idx

        self.mode = mode
        self.transform = transform

        self.samples = []

        self._build_index()

    def _build_index(self):
        for video_path in sorted(self.videos_path.iterdir()):
            if not video_path.is_dir():
                continue

            video_id = video_path.name

            if video_id not in self.split_ids:
                continue

            annotation_file = video_path / "annotations.txt"
            clip_categories = load_video_annot(annotation_file)

            for clip_path in sorted(video_path.iterdir()):
                if not clip_path.is_dir():
                    continue

                clip_id = clip_path.name
                tracking_file = self.annot_root / video_id / clip_id / f"{clip_id}.txt"

                if not tracking_file.exists():
                    continue

                frame_boxes = load_tracking_annot(tracking_file)
                scene_label = self.scene_to_idx[clip_categories[clip_id]]

                # B3 Stage 1
                # CNN on single player crop
                if self.mode == "person":
                    self._add_person_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label,
                    )

                # B3 Stage 2
                # 12 players from one frame
                elif self.mode == "person_grouped":
                    self._add_person_grouped_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label,
                    )

                # B1
                # middle frame only
                elif self.mode == "single_frame":
                    self._add_single_frame_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label,
                    )

                # ============================
                # B4 / B6
                # sequence of 9 images / no players
                elif self.mode == "clip_frames":
                    self._add_clip_samples(
                        video_id,
                        clip_id,
                        frame_boxes,
                        scene_label,
                    )

                # B5 / B7
                # player tracklets 12 players x 9 frames
                elif self.mode == "person_temporal":
                    self._add_person_temporal_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label,
                    )

        
                # backup only
                # old clip with players
                elif self.mode == "clip_with_players":
                    self.samples.append(
                        {
                            "video_id": video_id,
                            "clip_id": clip_id,
                            "clip_path": clip_path,
                            "frame_boxes": frame_boxes,
                            "scene_label": scene_label,
                        }
                    )
                else:
                    raise ValueError("Unknown mode")

    # B3 Stage 1
    def _add_person_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label,
    ):
        
        """ One sample = one player crop
            Output later:
            image -> player_label """
        for frame_id, boxes in frame_boxes.items():
            for box in boxes:
                self.samples.append(
                    {
                        "video_id": video_id,
                        "clip_id": clip_id,
                        "frame_id": frame_id,
                        "frame_path": clip_path / f"{frame_id}.jpg",
                        "box": box,
                        "player_label": box.category,
                        "scene_label": scene_label,
                    }
                )

    # B3 Stage 2
    def _add_person_grouped_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label,
    ):
        
        """ One sample = 12 players from ONE frame
            
            Used after B3 stage1:
            crop players -> CNN -> pooling -> scene
            
            Output later:
            images [12,C,H,W]  """
        # take first frame in clip
        # (because tracking annotation already keeps middle frames)
        frame_id = next(iter(frame_boxes))
        boxes = frame_boxes[frame_id]

        # Keep player order fixed
        # player 0 always stays player 0
        boxes = sorted(boxes, key=lambda x: x.player_ID)

        # cache labels
        if self.player_to_idx is not None:
            for box in boxes:
                box.label_idx = self.player_to_idx[box.category]

        self.samples.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_id": frame_id,
                "frame_path": clip_path / f"{frame_id}.jpg",
                "boxes": boxes,
                "scene_label": scene_label,
            }
        )

  
    # B1
    # Image classification
    def _add_single_frame_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label,
    ):
        """  Only middle frame
            Full image
            
            Output later:
            image [C,H,W] """
        
        frame_ids = sorted(frame_boxes.keys())
        middle_idx = len(frame_ids) // 2
        frame_id = frame_ids[middle_idx]

        self.samples.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_id": frame_id,
                "frame_path": clip_path / f"{frame_id}.jpg",
                "scene_label": scene_label,
            }
        )


    # B4 / B6
    # Temporal image model
    def _add_clip_samples(self, video_id, clip_id, frame_boxes, scene_label):
        """ One sample = one clip
            
            Output later:
            frames [T,C,H,W]
            
            T = 9 """
        
        self.samples.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_boxes": frame_boxes,
                "scene_label": scene_label,
            }
        )

    # B5 / B7
    # Temporal player model
    def _add_person_temporal_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label,
    ):
        
        """We keep every player's trajectory:
            player0:
              frame1 crop
              frame2 crop
              ...
              frame9 crop
            player1:
              frame1 crop
            Output later:
            images [12,9,C,H,W] """
            
            
        frame_ids = sorted(frame_boxes.keys())

        # organize:player_id to player_id
        player_tracks = {idx: [] for idx in range(12)}

        for frame_id in frame_ids:
            boxes = frame_boxes[frame_id]

            # keep player order
            boxes = sorted(boxes, key=lambda x: x.player_ID)

            for box in boxes:
                player_tracks[box.player_ID].append(
                    {
                        "frame_id": frame_id,
                        "box": box,
                        "label_idx": self.player_to_idx[box.category]
                    }
            )

        self.samples.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "clip_path": clip_path,
                "player_tracks": player_tracks,
                "scene_label": scene_label,
            }
        )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _crop_player(self, image, box: BoxInfo):
        x1, y1, x2, y2 = box.box
        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, index):
        sample = self.samples[index]

  
        # B3 Stage 1
        # Input:
        # one player crop
        # Output:
        # image:
        # [C,H,W]
        # player_label
        # =====================================================
        if self.mode == "person":
            image = self._load_image(sample["frame_path"])
            image = self._crop_player(image, sample["box"])

            if self.transform:
                image = self.transform(image)

            return {
                "image": image,
                "player_label": self.player_to_idx[sample["player_label"]],
                "scene_label": sample["scene_label"],
                "video_id": sample["video_id"],
                "clip_id": sample["clip_id"],
                "frame_id": sample["frame_id"],
            }
            
        # B3 Stage 2

        elif self.mode == "person_grouped":
            image = self._load_image(sample["frame_path"])
            boxes = sample["boxes"]

            player_images = []
            player_labels = []

            for box in boxes:
                crop = self._crop_player(image, box)

                if self.transform:
                    crop = self.transform(crop)

                player_images.append(crop)
                player_labels.append(box.label_idx)

            # padding missing players
            while len(player_images) < 12:
                if player_images:
                    pad = torch.zeros_like(player_images[0])
                else:
                    pad = torch.zeros(3, 224, 224)

                player_images.append(pad)
                player_labels.append(-1)

            player_images = torch.stack(player_images)
            player_labels = torch.tensor(player_labels, dtype=torch.long)

            return {
                "images": player_images,
                "player_labels": player_labels,
                "scene_label": sample["scene_label"],
            }

      
        # B1
        # Full image
        # middle frame only
        # Output:
        # image [C,H,W]
        elif self.mode == "single_frame":
            image = self._load_image(sample["frame_path"])

            if self.transform:
                image = self.transform(image)

            return {
                "image": image,
                "scene_label": sample["scene_label"],
            }


        # B4 / B6
        elif self.mode == "clip_frames":
            frames = []
            frame_ids = sorted(sample["frame_boxes"].keys())

            for frame_id in frame_ids:
                image_path = (
                    self.videos_path / sample["video_id"] / sample["clip_id"] / f"{frame_id}.jpg"
                )
                image = self._load_image(image_path)

                if self.transform:
                    image = self.transform(image)

                frames.append(image)

            return {
                "frames": torch.stack(frames),
                "scene_label": sample["scene_label"],
            }

        
       
        # B5 stageA / B7
        # Temporal player model
        elif self.mode == "person_temporal":

            player_tracks = sample["player_tracks"]

            # --------------------------------------------------
            # Prepare storage for 12 players
            # Each player will contain 9 frames
            # --------------------------------------------------
            all_players = [[] for _ in range(12)]
            player_labels = []

            # Get player labels
            for player_id in range(12):

                track = player_tracks[player_id]

                if track:
                    player_label = track[0]["label_idx"]
                else:
                    player_label = -1

                player_labels.append(player_label)

            # --------------------------------------------------
            # Load each frame ONCE
            # --------------------------------------------------
            frame_ids = sorted(
                {
                    item["frame_id"]
                    for player_id in range(12)
                    for item in player_tracks[player_id]
                }
            )

            for frame_id in frame_ids:

                image_path = sample["clip_path"] / f"{frame_id}.jpg"

                # Load image only once
                image = self._load_image(image_path)

                # --------------------------------------------------
                # Crop all players from this frame
                # --------------------------------------------------
                for player_id in range(12):

                    track = player_tracks[player_id]

                    # Find this player's box for this frame
                    box = None

                    for item in track:
                        if item["frame_id"] == frame_id:
                            box = item["box"]
                            break

                    # Player exists in this frame
                    if box is not None:

                        crop = self._crop_player(
                            image,
                            box
                        )

                        if self.transform:
                            crop = self.transform(crop)

                        all_players[player_id].append(crop)

            # --------------------------------------------------
            # Ensure every player has exactly 9 frames
            # --------------------------------------------------
            for player_id in range(12):

                player_frames = all_players[player_id]

                # Missing player completely
                if len(player_frames) == 0:

                    if all_players[0]:
                        pad = torch.zeros_like(
                            all_players[0][0]
                        )
                    else:
                        pad = torch.zeros(
                            3,
                            224,
                            224
                        )

                    player_frames = [
                        pad.clone()
                        for _ in range(9)
                    ]

                # Less than 9 frames
                elif len(player_frames) < 9:

                    pad = torch.zeros_like(
                        player_frames[0]
                    )

                    while len(player_frames) < 9:
                        player_frames.append(
                            pad.clone()
                        )

                # More than 9 frames
                elif len(player_frames) > 9:

                    player_frames = player_frames[:9]

                all_players[player_id] = torch.stack(
                    player_frames
                )

            # --------------------------------------------------
            # Final shape:
            # (12, 9, C, H, W)
            # --------------------------------------------------
            all_players = torch.stack(
                all_players
            )

            player_labels = torch.tensor(
                player_labels,
                dtype=torch.long
            )

            return {
                "images": all_players,
                "player_labels": player_labels,
                "scene_label": sample["scene_label"],
            }




# B1  -> single_frame
# B2  -> person
# B3A -> person
# B3B -> person_grouped

# B4  -> clip_frames

# B5  -> person_temporal

# B6  -> clip_frames

# B7  -> person_temporal + frame LSTM
# B8  -> person_temporal + team pooling



# ============================================================
# Dataset Modes & Input Shapes per Baseline
# ============================================================

# B1 -> single_frame
# Input:
#   (B, C, H, W)
# Example:
#   (16, 3, 224, 224)
#
# Each sample = single frame
# Model:
#   ResNet50 -> Classifier
#
# Output:
#   (B, num_classes)


# ------------------------------------------------------------


# B2 -> person
# Input:
#   (B, P, C, H, W)
# Example:
#   (16, 12, 3, 224, 224)
#
# Each sample = one clip with 12 players
# Model:
#   ResNet50 per player
#   Concatenate player features
#   MLP classifier
#
# Output:
#   (B, num_classes)


# ------------------------------------------------------------


# B3A -> person
# Input:
#   (B, C, H, W)
# Example:
#   (16, 3, 224, 224)
#
# Each sample = single player crop
# Task:
#   Person action classification
#
# Model:
#   ResNet50 -> 9 person classes
#
# Output:
#   (B, 9)


# ------------------------------------------------------------


# B3B -> person_grouped
# Input:
#   (B, P, C, H, W)
# Example:
#   (16, 12, 3, 224, 224)
#
# Each sample = group of players
#
# Backbone:
#   B3A trained ResNet50
#
# Flow:
#   Player images
#       |
#   CNN features
#       |
#   Player pooling
#       |
#   Group classifier
#
# Output:
#   (B, 8)


# ------------------------------------------------------------


# B4 -> clip_frames
# Input:
#   (B, T, C, H, W)
# Example:
#   (16, 9, 3, 224, 224)
#
# Each sample = temporal clip
# T = number of frames
#
# Model:
#   B1 Backbone
#       |
#   Frame features
#       |
#   LSTM
#       |
#   Classifier
#
# Output:
#   (B, 8)


# ------------------------------------------------------------


# B5 -> person_temporal
# Input:
#   (B, T, P, C, H, W)
# Example:
#   (16, 9, 12, 3, 224, 224)
#
# Each sample:
#   9 frames
#   12 players per frame
#
# Model:
#   Person CNN
#       |
#   Temporal modeling
#       |
#   Classifier
#
# Output:
#   (B, 8)


# ------------------------------------------------------------


# B6 -> clip_frames
# Input:
#   (B, T, C, H, W)
# Example:
#   (16, 9, 3, 224, 224)
#
# Each sample = frame sequence
#
# Model:
#   CNN feature extractor
#       |
#   Temporal aggregation
#
# Output:
#   (B, 8)


# ------------------------------------------------------------


# B7 -> person_temporal + frame LSTM
# Input:
#   (B, T, P, C, H, W)
# Example:
#   (16, 9, 12, 3, 224, 224)
#
# Flow:
#   Each frame:
#       players
#          |
#       CNN
#          |
#     player features
#
#   Frame-level LSTM
#
# Output:
#   (B, 8)


# ------------------------------------------------------------


# B8 -> person_temporal + team pooling
# Input:
#   (B, T, P, C, H, W)
# Example:
#   (16, 9, 12, 3, 224, 224)
#
# Flow:
#   Player features
#       |
#   Team/Player pooling
#       |
#   Temporal model
#       |
#   Classifier
#
# Output:
#   (B, 8)

# ============================================================