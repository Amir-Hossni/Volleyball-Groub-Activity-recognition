from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from .volleyball_annot_loader import load_tracking_annot, load_video_annot

from .boxinfo import BoxInfo



class VolleyballDatasetv2(Dataset):

    def __init__(
        self,
        videos_path,
        annot_root,
        split_ids,
        scene_to_idx,
        player_to_idx=None,
        mode="clip",
        transform=None
    ):

        self.videos_path = Path(videos_path)
        self.annot_root = Path(annot_root)

        self.split_ids = (
            set(split_ids)
            if not isinstance(split_ids, set)
            else split_ids
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
                tracking_file = ( self.annot_root / video_id / clip_id / f"{clip_id}.txt")

                if not tracking_file.exists():
                    continue


                frame_boxes = load_tracking_annot(tracking_file)
                scene_label = self.scene_to_idx[clip_categories[clip_id]]


                if self.mode == "person":

                    self._add_person_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )


                elif self.mode == "person_grouped":

                    self._add_person_grouped_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )


                elif self.mode == "frame":

                    self._add_frame_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )


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
                        "mode must be person, "
                        "person_grouped, "
                        "frame or clip"
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
                        "frame_path": clip_path / f"{frame_id}.jpg",
                        "box": box,
                        "player_label": box.category,
                        "scene_label": scene_label
                    }
                )



    def _add_person_grouped_samples(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label
    ):

        frame_id = next(iter(frame_boxes))
        boxes = frame_boxes[frame_id]

        # Pre-sort boxes by player_ID once during build
        # instead of sorting on every __getitem__ call
        sorted_boxes = sorted(
            boxes,
            key=lambda x: x.player_ID
        )

        # Pre-resolve player label strings to integer indices
        # to avoid dict lookups inside the hot __getitem__ path.
        for box in sorted_boxes:
            box.label_idx = self.player_to_idx[box.category]

        self.samples.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_id": frame_id,
                "frame_path": clip_path / f"{frame_id}.jpg",
                "boxes": sorted_boxes,
                "scene_label": scene_label
            }
        )



    def _add_frame_samples_temp(
        self,
        video_id,
        clip_id,
        clip_path,
        frame_boxes,
        scene_label
    ):

        for frame_id, boxes in frame_boxes.items():

            self.samples.append(
                {
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "frame_id": frame_id,
                    "frame_path": clip_path / f"{frame_id}.jpg",
                    "boxes": boxes,
                    "scene_label": scene_label
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

        frame_ids = sorted(frame_boxes.keys())

        middle_idx = len(frame_ids) // 2

        frame_id = frame_ids[middle_idx]


        self.samples.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_id": frame_id,
                "frame_path": clip_path / f"{frame_id}.jpg",
                "scene_label": scene_label
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

        # Person level
        # B2 / B3
        # ======================
        if self.mode == "person":

            image = self._load_image(sample["frame_path"])

            image = self._crop_player(image, sample["box"])

            if self.transform:
                image = self.transform(image)

            return {

                "image": image,

                "player_label": self.player_to_idx[sample["player_label"]],

                "scene_label": sample["scene_label"],

                "bbox": sample["box"].box,

                "player_id": sample["box"].player_ID,

                "video_id": sample["video_id"],

                "clip_id": sample["clip_id"],

                "frame_id": sample["frame_id"]
            }



        # Person Grouped level
        # ======================
        elif self.mode == "person_grouped":

            image = self._load_image(sample["frame_path"])

            # Boxes are already pre-sorted by player_ID from
            # _build_index and contain pre-resolved label_idx.
            boxes = sample["boxes"]
            num_boxes = len(boxes)

            # Pre-allocate lists for the fixed 12-player maximum
            player_images = [None] * 12
            player_labels = [None] * 12

            # Shared zero tensor for padding missing players
            pad_tensor = None

            if self.transform:

                for i in range(num_boxes):

                    box = boxes[i]
                    crop = self._crop_player(image, box)

                    crop = self.transform(crop)
                    player_images[i] = crop
                    player_labels[i] = box.label_idx

                # Pad with zero tensors for missing players
                for i in range(num_boxes, 12):

                    if pad_tensor is None:

                        # Create a zero tensor with the same shape
                        # as a transformed crop by applying
                        # transform to a dummy crop.
                        dummy_box = boxes[0]
                        dummy_crop = self._crop_player(image, dummy_box)

                        pad_tensor = (self.transform(dummy_crop) * 0.0)

                    player_images[i] = pad_tensor.clone()
                    player_labels[i] = -1

            else:

                for i in range(num_boxes):

                    box = boxes[i]
                    crop = self._crop_player(image, box)

                    player_images[i] = crop
                    player_labels[i] = box.label_idx

                # Pad with zero tensors for missing players
                for i in range(num_boxes, 12):

                    if pad_tensor is None:
                        pad_tensor = torch.zeros_like(player_images[0])

                    player_images[i] = pad_tensor.clone()
                    player_labels[i] = -1

            player_images = torch.stack(player_images)
            player_labels = torch.tensor(player_labels, dtype=torch.long)

            return {
                "images": player_images,
                
                "player_labels": player_labels,

                "scene_label": sample["scene_label"],

                "video_id": sample["video_id"],

                "clip_id": sample["clip_id"],

                "frame_id": sample["frame_id"]
            }


        # Frame level
        # B1
        # ======================
        elif self.mode == "frame":

            image = self._load_image(sample["frame_path"])


            if self.transform:
                image = self.transform(image)


            return {

                "image": image,

                "scene_label": sample["scene_label"],

                "video_id": sample["video_id"],

                "clip_id": sample["clip_id"],

                "frame_id": sample["frame_id"]
            }

        # Clip level
        else:
            frames = []
            players = []


            for frame_id, boxes in sample["frame_boxes"].items():

                image_path = (sample["clip_path"] / f"{frame_id}.jpg")

                image = self._load_image(image_path)

                if self.transform:
                    image = self.transform(image)

                frames.append(image)
                players.append(boxes)


            return {

                "frames": torch.stack(frames),

                # "players": players,

                "scene_label": sample["scene_label"]
            }