from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from .volleyball_annot_loader import (
    load_tracking_annot,
    load_video_annot
)

from .boxinfo import BoxInfo



class VolleyballDataset(Dataset):

    def __init__(
        self,
        videos_path,
        annot_root,
        split_ids,
        scene_to_idx,
        mode="clip",
        transform=None
    ):

        self.videos_path = Path(videos_path)
        self.annot_root = Path(annot_root)

        self.split_ids = split_ids
        self.scene_to_idx = scene_to_idx

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

                tracking_file = (self.annot_root/video_id/clip_id/f"{clip_id}.txt")

                if not tracking_file.exists():
                    continue

                frame_boxes = load_tracking_annot(tracking_file)


                scene_label = self.scene_to_idx[clip_categories[clip_id]]

                if self.mode == "person":
                    self._add_person_samples(video_id, clip_id, clip_path, frame_boxes, scene_label)

                elif self.mode == "frame":
                    self._add_frame_samples(video_id, clip_id, clip_path, frame_boxes,scene_label)

                elif self.mode == "clip":
                    self.samples.append(
                        {
                            "video_id": video_id,
                            "clip_id": clip_id,
                            "clip_path": clip_path,
                            "frame_boxes": frame_boxes,
                            "scene_label": scene_label
                        }
                    )


                else:

                    raise ValueError(
                        "mode must be person, frame or clip"
                    )



    def _add_person_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label
    ):


        for frame_id, boxes in frame_boxes.items():

            for box in boxes:

                self.samples.append(
                    {
                        "video_id": video_id,
                        "clip_id": clip_id,
                        "frame_id": frame_id,

                        "frame_path":
                            clip_path / f"{frame_id}.jpg",

                        "box": box,

                        "player_label":
                            box.category,

                        "scene_label":
                            scene_label
                    }
                )



    def _add_frame_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label
    ):


        for frame_id in frame_boxes.keys():

            self.samples.append(
                {
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "frame_id": frame_id,

                    "frame_path":
                        clip_path / f"{frame_id}.jpg",

                    "scene_label":
                        scene_label
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



        # ======================
        # Person level
        # B2 / B3
        # ======================

        if self.mode == "person":


            image = self._load_image(
                sample["frame_path"]
            )


            image = self._crop_player(
                image,
                sample["box"]
            )


            if self.transform:
                image = self.transform(image)



            return {

                "image": image,

                "player_label":
                    sample["player_label"],

                "scene_label":
                    sample["scene_label"],

                "bbox":
                    sample["box"].box,

                "player_id":
                    sample["box"].player_ID,

                "video_id":
                    sample["video_id"],

                "clip_id":
                    sample["clip_id"],

                "frame_id":
                    sample["frame_id"]
            }




        # ======================
        # Frame level
        # B1
        # ======================

        elif self.mode == "frame":


            image = self._load_image(
                sample["frame_path"]
            )


            if self.transform:
                image = self.transform(image)



            return {

                "image": image,

                "scene_label":
                    sample["scene_label"]
            }




        # ======================
        # Clip level
        # ======================

        else:


            frames = []
            players = []


            for frame_id, boxes in sample["frame_boxes"].items():


                image_path = (
                    sample["clip_path"]
                    /
                    f"{frame_id}.jpg"
                )


                image = self._load_image(
                    image_path
                )


                if self.transform:
                    image = self.transform(image)


                frames.append(image)

                players.append(boxes)



            return {

                "frames":
                    torch.stack(frames),

                "players":
                    players,

                "scene_label":
                    sample["scene_label"]
            }