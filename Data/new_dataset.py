 
from pathlib import Path
import pickle

import torch
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from PIL import Image


class VolleyballDatasetN(Dataset):

    _MODES = ("person", "frame", "clip", "clip_crops")

    def __init__(
        self,
        videos_path,
        pkl_path,
        split_ids,
        scene_to_idx,
        player_to_idx=None,
        mode="clip",
        transform=None,
        max_players=12,
        return_meta=False,
    ):
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}")

        if mode in ("person", "clip_crops") and player_to_idx is None:
            raise ValueError(
                f"mode='{mode}' requires player_to_idx mapping"
            )

        self.videos_path = Path(videos_path)
        self.split_ids = set(split_ids)
        self.scene_to_idx = scene_to_idx
        self.player_to_idx = player_to_idx
        self.mode = mode
        self.transform = transform
        self.max_players = max_players
        self.return_meta = return_meta

        with open(pkl_path, "rb") as f:
            self.annotations = pickle.load(f)

        self.samples = []
        self._build_index()

        self.annotations = None


    def _frame_path(self, video_id, clip_id, frame_id) -> str:
        return str(
            self.videos_path / video_id / clip_id / f"{frame_id}.jpg"
        )


    @staticmethod
    def _sorted_boxes(boxes):
        """Stable player order (by player_ID); plain tuples, no dicts."""
        return [
            (tuple(b["box"]), b["category"], b["player_ID"])
            for b in sorted(boxes, key=lambda b: b["player_ID"])
        ]


    def _build_index(self):

        for video_id, clips in self.annotations.items():

            if video_id not in self.split_ids:
                continue

            for clip_id, clip_data in clips.items():

                scene_label = self.scene_to_idx[clip_data["category"]]
                frame_boxes = clip_data["frame_boxes_dct"]


                if self.mode in ("person", "frame"):

                    for frame_id, boxes in frame_boxes.items():

                        if self.mode == "person" and not boxes:
                            continue

                        self.samples.append(
                            {
                                "path": self._frame_path(
                                    video_id,
                                    clip_id,
                                    frame_id
                                ),
                                "boxes": self._sorted_boxes(boxes),
                                "scene": scene_label,
                                "meta": (
                                    video_id,
                                    clip_id,
                                    str(frame_id)
                                ),
                            }
                        )


                elif self.mode == "clip":

                    frame_ids = sorted(frame_boxes)

                    self.samples.append(
                        {
                            "paths": [
                                self._frame_path(
                                    video_id,
                                    clip_id,
                                    fid
                                )
                                for fid in frame_ids
                            ],
                            "scene": scene_label,
                            "meta": (
                                video_id,
                                clip_id,
                                ""
                            ),
                        }
                    )


                else:  # clip_crops

                    frame_ids = sorted(frame_boxes)

                    per_frame = [
                        {
                            b["player_ID"]:
                            (
                                tuple(b["box"]),
                                b["category"]
                            )
                            for b in frame_boxes[fid]
                        }
                        for fid in frame_ids
                    ]

                    common = set(per_frame[0])

                    for pf in per_frame[1:]:
                        common &= set(pf)

                    track_ids = sorted(common)[: self.max_players]

                    if not track_ids:
                        continue


                    self.samples.append(
                        {
                            "paths": [
                                self._frame_path(
                                    video_id,
                                    clip_id,
                                    fid
                                )
                                for fid in frame_ids
                            ],

                            "tracks": [
                                [
                                    per_frame[t][pid][0]
                                    for pid in track_ids
                                ]
                                for t in range(len(frame_ids))
                            ],

                            "labels": [
                                self.player_to_idx[
                                    per_frame[0][pid][1]
                                ]
                                for pid in track_ids
                            ],

                            "scene": scene_label,

                            "meta": (
                                video_id,
                                clip_id,
                                ""
                            ),
                        }
                    )                
    
        def __len__(self):
            return len(self.samples)


        @staticmethod
        def _decode(path):
            return Image.open(path).convert("RGB")


        def _prep(self, pil_image):
            if self.transform:
                return self.transform(pil_image)

            return TF.to_tensor(pil_image)


        def _pad_players(self, crops, labels):
            """
            Pad crops/labels to max_players.
            Returns:
            crops  -> (max_players, C, H, W)
            labels -> (max_players,)
            mask   -> (max_players,)
            """

            P = len(crops)

            mask = torch.zeros(
                self.max_players,
                dtype=torch.bool
            )

            mask[:P] = True

            pad = self.max_players - P

            if pad > 0:

                zero = torch.zeros_like(crops[0])

                crops = crops + [zero] * pad
                labels = labels + [-1] * pad

            return (
                torch.stack(crops),
                torch.tensor(labels, dtype=torch.long),
                mask
            )


    def __getitem__(self, index):

        s = self.samples[index]


        if self.mode == "person":

            image = self._decode(s["path"])

            boxes = s["boxes"][: self.max_players]


            crops = [
                self._prep(image.crop(box))
                for box, _, _ in boxes
            ]


            labels = [
                self.player_to_idx[cat]
                for _, cat, _ in boxes
            ]


            crops, labels, mask = self._pad_players(
                crops,
                labels
            )


            out = (
                crops,
                labels,
                mask,
                s["scene"]
            )


        elif self.mode == "frame":

            out = (
                self._prep(self._decode(s["path"])),
                s["scene"]
            )


        elif self.mode == "clip":

            frames = torch.stack(
                [
                    self._prep(self._decode(p))
                    for p in s["paths"]
                ]
            )

            out = (
                frames,
                s["scene"]
            )


        else:  # clip_crops

            per_frame = []

            for t, path in enumerate(s["paths"]):

                image = self._decode(path)

                per_frame.append(
                    torch.stack(
                        [
                            self._prep(image.crop(box))
                            for box in s["tracks"][t]
                        ]
                    )
                )


            crops = torch.stack(per_frame)
            # (T, P, C, H, W)


            P = crops.shape[1]


            mask = torch.zeros(
                self.max_players,
                dtype=torch.bool
            )

            mask[:P] = True


            labels = torch.full(
                (self.max_players,),
                -1,
                dtype=torch.long
            )


            labels[:P] = torch.tensor(
                s["labels"],
                dtype=torch.long
            )


            if P < self.max_players:

                pad_shape = (
                    crops.shape[0],
                    self.max_players - P,
                    *crops.shape[2:]
                )

                crops = torch.cat(
                    [
                        crops,
                        crops.new_zeros(pad_shape)
                    ],
                    dim=1
                )


            out = (
                crops,
                labels,
                mask,
                s["scene"]
            )


        return (
            out + (s["meta"],)
            if self.return_meta
            else out
        )



def flatten_person_batch(crops, labels, mask):
    """
    (B,P,C,H,W), (B,P), (B,P)
        ->
    (N,C,H,W), (N,)

    Remove padded players.
    """

    B, P = mask.shape

    flat_mask = mask.reshape(B * P)


    return (
        crops.reshape(B * P, *crops.shape[2:])[flat_mask],
        labels.reshape(B * P)[flat_mask]
    )



# Usage for person classifier training

# ds = VolleyballDataset(
#     videos_path,
#     pkl_path,
#     TRAIN_IDS,
#     SCENE_TO_IDX,
#     player_to_idx=PLAYER_TO_IDX,
#     mode="person",
#     transform=tf
# )


# loader = DataLoader(
#     ds,
#     batch_size=16,
#     shuffle=True,
#     num_workers=4,
#     persistent_workers=True,
#     prefetch_factor=4,
#     pin_memory=True
# )


# for crops, labels, mask, _scene in loader:

#     x, y = flatten_person_batch(
#         crops,
#         labels,
#         mask
#     )

#     loss = criterion(
#         model(x.to(device)),
#         y.to(device)
#     )                