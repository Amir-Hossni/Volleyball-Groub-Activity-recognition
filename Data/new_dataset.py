
from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset
from PIL import Image

from .volleyball_annot_loader import load_tracking_annot

class VolleyballDatasetDirect(Dataset):

    def __init__(
        self,
        data,
        scene_to_idx,
        player_to_idx,
        transform=None
    ):

        self.data = data

        self.scene_to_idx = scene_to_idx
        self.player_to_idx = player_to_idx

        self.transform = transform

        self.samples = []

        self._build_index()


    def _build_index(self):

        for sample in self.data:

            video_id = sample["video_id"]
            clip_id = sample["clip_id"]

            scene_label = self.scene_to_idx[
                sample["group_label"]
            ]


            # Read tracking file directly
            frame_boxes_dct = load_tracking_annot(
                sample["tracking_path"]
            )


            # Same logic as your person_grouped
            for frame_id, boxes in frame_boxes_dct.items():


                # sort players once
                sorted_boxes = sorted(
                    boxes,
                    key=lambda x: x.player_ID
                )


                # convert label string to index
                for box in sorted_boxes:

                    box.label_idx = self.player_to_idx[
                        box.category
                    ]


                self.samples.append(
                    {
                        "video_id": video_id,

                        "clip_id": clip_id,

                        "frame_id": frame_id,

                        "image_path":
                            sample["image_path"],

                        "boxes": sorted_boxes,

                        "scene_label": scene_label
                    }
                )


    def __len__(self):

        return len(self.samples)



    def _load_image(self, path):

        return Image.open(path).convert("RGB")



    def _crop_player(self, image, box):

        x1, y1, x2, y2 = box.box

        return image.crop(
            (x1, y1, x2, y2)
        )



    def __getitem__(self, index):

        sample = self.samples[index]


        image = self._load_image(
            sample["image_path"]
        )


        boxes = sample["boxes"]


        player_images = []
        player_labels = []


        for box in boxes[:12]:

            crop = self._crop_player(
                image,
                box
            )


            if self.transform:

                crop = self.transform(crop)


            player_images.append(crop)

            player_labels.append(
                box.label_idx
            )


        # padding
        while len(player_images) < 12:


            if self.transform:

                dummy = torch.zeros(
                    3,
                    224,
                    224
                )

            else:

                dummy = torch.zeros_like(
                    player_images[0]
                )


            player_images.append(dummy)

            player_labels.append(-1)



        player_images = torch.stack(
            player_images
        )


        player_labels = torch.tensor(
            player_labels,
            dtype=torch.long
        )


        
        
        return {

            "images":
                player_images,

            "player_labels":
                player_labels,

            "scene_label":
                sample["scene_label"],

            "video_id":
                sample["video_id"],

            "clip_id":
                sample["clip_id"],

            "frame_id":
                sample["frame_id"]
        }