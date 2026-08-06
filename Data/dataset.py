from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from .volleyball_annot_loader import load_tracking_annot, load_video_annot
from .boxinfo import BoxInfo



class VolleyballDataset(Dataset):

    def __init__(
        self,
        videos_path,
        annot_root,
        split_ids,
        scene_to_idx,
        player_to_idx=None,
        mode="clip_frames",
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


                tracking_file = (
                    self.annot_root
                    /
                    video_id
                    /
                    clip_id
                    /
                    f"{clip_id}.txt"
                )


                if not tracking_file.exists():
                    continue



                frame_boxes = load_tracking_annot(
                    tracking_file
                )


                scene_label = self.scene_to_idx[
                    clip_categories[clip_id]
                ]



                # ============================
                # B3 Stage 1
                # CNN on single player crop
                # ============================
                if self.mode == "person":

                    self._add_person_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )



                # ============================
                # B3 Stage 2
                # 12 players from one frame
                # ============================
                elif self.mode == "person_grouped":

                    self._add_person_grouped_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )



                # ============================
                # B1
                # middle frame only
                # ============================
                elif self.mode == "single_frame":

                    self._add_single_frame_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )



                # ============================
                # B4 / B6
                # sequence of 9 images
                # no players
                # ============================
                elif self.mode == "clip_frames":

                    self._add_clip_samples(
                        video_id,
                        clip_id,
                        frame_boxes,
                        scene_label
                    )



                # ============================
                # B5 / B7
                # player tracklets
                # 12 players x 9 frames
                # ============================
                elif self.mode == "person_temporal":

                    self._add_person_temporal_samples(
                        video_id,
                        clip_id,
                        clip_path,
                        frame_boxes,
                        scene_label
                    )



                # ============================
                # backup only
                # old clip with players
                # ============================
                elif self.mode == "clip_with_players":

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
                        "Unknown mode"
                    )
        # =====================================================
        # B3 Stage 1
        # One sample = one player crop
        # Output later:
        # image -> player_label
        # =====================================================
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



        # =====================================================
        # B3 Stage 2
        # One sample = 12 players from ONE frame
        #
        # Used after B3 stage1:
        # crop players -> CNN -> pooling -> scene
        #
        # Output later:
        # images [12,C,H,W]
        # =====================================================
        def _add_person_grouped_samples(
            self,
            video_id,
            clip_id,
            clip_path,
            frame_boxes,
            scene_label
        ):


            # take first frame in clip
            # (because tracking annotation already keeps middle frames)
            frame_id = next(iter(frame_boxes))


            boxes = frame_boxes[frame_id]


            # Keep player order fixed
            # player 0 always stays player 0
            boxes = sorted(
                boxes,
                key=lambda x: x.player_ID
            )


            # cache labels
            if self.player_to_idx is not None:

                for box in boxes:

                    box.label_idx = (
                        self.player_to_idx[box.category]
                    )


            self.samples.append(
                {
                    "video_id": video_id,

                    "clip_id": clip_id,

                    "frame_id": frame_id,

                    "frame_path":
                        clip_path / f"{frame_id}.jpg",

                    "boxes": boxes,

                    "scene_label":
                        scene_label
                }
            )



        # =====================================================
        # B1
        # Image classification
        #
        # Only middle frame
        # Full image
        #
        # Output later:
        # image [C,H,W]
        # =====================================================
        def _add_single_frame_samples(
            self,
            video_id,
            clip_id,
            clip_path,
            frame_boxes,
            scene_label
        ):


            frame_ids = sorted(
                frame_boxes.keys()
            )


            middle_idx = len(frame_ids) // 2


            frame_id = frame_ids[middle_idx]


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



        # =====================================================
        # B4 / B6
        # Temporal image model
        #
        # One sample = one clip
        #
        # Output later:
        # frames [T,C,H,W]
        #
        # T = 9
        # =====================================================
        def _add_clip_samples(
            self,
            video_id,
            clip_id,
            frame_boxes,
            scene_label
        ):


            self.samples.append(
                {
                    "video_id": video_id,

                    "clip_id": clip_id,

                    "frame_boxes": frame_boxes,

                    "scene_label":
                        scene_label
                }
            )



        # =====================================================
        # B5 / B7
        # NEW
        #
        # One sample = one clip
        #
        # We keep every player's trajectory:
        #
        # player0:
        #   frame1 crop
        #   frame2 crop
        #   ...
        #   frame9 crop
        #
        # player1:
        #   frame1 crop
        #   ...
        #
        # Output later:
        # images [12,9,C,H,W]
        # =====================================================
        def _add_person_temporal_samples(
            self,
            video_id,
            clip_id,
            clip_path,
            frame_boxes,
            scene_label
        ):


            frame_ids = sorted(
                frame_boxes.keys()
            )


            # organize:
            #
            # player_id
            #       |
            #       list of frames
            #
            player_tracks = {
                idx: []
                for idx in range(12)
            }



            for frame_id in frame_ids:

                boxes = frame_boxes[frame_id]


                # keep player order
                boxes = sorted(
                    boxes,
                    key=lambda x: x.player_ID
                )


                for box in boxes:

                    player_tracks[
                        box.player_ID
                    ].append(
                        {
                            "frame_id": frame_id,
                            "box": box
                        }
                    )



            self.samples.append(
                {
                    "video_id": video_id,

                    "clip_id": clip_id,

                    "clip_path": clip_path,

                    "player_tracks": player_tracks,

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

        return image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )



    def __getitem__(self, index):

        sample = self.samples[index]



        # =====================================================
        # B3 Stage 1
        #
        # Input:
        # one player crop
        #
        # Output:
        # image:
        # [C,H,W]
        #
        # player_label
        # =====================================================
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

                "image":
                    image,


                "player_label":
                    self.player_to_idx[
                        sample["player_label"]
                    ],


                "scene_label":
                    sample["scene_label"],


                "video_id":
                    sample["video_id"],


                "clip_id":
                    sample["clip_id"],


                "frame_id":
                    sample["frame_id"]

            }



        # =====================================================
        # B3 Stage 2
        #
        # Input:
        # 12 players from one frame
        #
        # Output:
        #
        # images:
        # [12,C,H,W]
        #
        # player_labels:
        # [12]
        #
        # =====================================================
        elif self.mode == "person_grouped":


            image = self._load_image(
                sample["frame_path"]
            )


            boxes = sample["boxes"]


            player_images = []

            player_labels = []



            for box in boxes:


                crop = self._crop_player(
                    image,
                    box
                )


                if self.transform:

                    crop = self.transform(crop)



                player_images.append(
                    crop
                )


                player_labels.append(
                    box.label_idx
                )



            # padding missing players
            while len(player_images) < 12:


                pad = torch.zeros_like(
                    player_images[0]
                )


                player_images.append(
                    pad
                )


                player_labels.append(
                    -1
                )



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
                    sample["scene_label"]

            }



        # =====================================================
        # B1
        #
        # Full image
        # middle frame only
        #
        # Output:
        # image [C,H,W]
        #
        # =====================================================
        elif self.mode == "single_frame":


            image = self._load_image(
                sample["frame_path"]
            )


            if self.transform:

                image = self.transform(image)



            return {

                "image":
                    image,


                "scene_label":
                    sample["scene_label"]

            }  
        
                # =====================================================
        # B4 / B6
        # B4:
        # CNN -> features -> LSTM
        #
        # =====================================================
        elif self.mode == "clip_frames":


            frames = []


            frame_ids = sorted(
                sample["frame_boxes"].keys()
            )


            for frame_id in frame_ids:


                image_path = (
                    self.videos_path
                    /
                    sample["video_id"]
                    /
                    sample["clip_id"]
                    /
                    f"{frame_id}.jpg"
                )


                image = self._load_image(
                    image_path
                )


                if self.transform:
                    image = self.transform(image)

                frames.append(image)


            return {
                "frames": torch.stack(frames),
                "scene_label": sample["scene_label"]
                }

        # B5 / B7
        elif self.mode == "person_temporal":

            player_tracks = sample["player_tracks"]

            all_players = []

            for player_id in range(12):

                track = player_tracks[player_id]
                player_frames = []

                for item in track:

                    frame_id = item["frame_id"]

                    box = item["box"]

                    image_path = (sample["clip_path"] / f"{frame_id}.jpg")

                    image = self._load_image(image_path)

                    crop = self._crop_player(image, box)

                    if self.transform:
                        crop = self.transform(crop)

                    player_frames.append(crop)



                # Handle missing player
                # Example:
                # player disappeared
                # We create zero sequence
                # ==========================================

                if len(player_frames)==0:

                    if len(all_players)>0:

                        dummy = torch.zeros_like(all_players[0][0])
                    else:
                        # fallback
                        dummy = torch.zeros(3, 224, 224)

                    player_frames = [dummy for _ in range(9)]


                # Ensure fixed temporal length
                # Dataset gives 9 frames
                # But safe handling:
                # ==========================================
                if len(player_frames) < 9:

                    pad = torch.zeros_like(player_frames[0])

                    while len(player_frames)<9:

                        player_frames.append(pad.clone())

                elif len(player_frames)>9:

                    player_frames = (player_frames[:9])

                player_frames = torch.stack(player_frames)

                all_players.append(player_frames)

            all_players = torch.stack(all_players)

            return {
                "images": all_players,
                "scene_label": sample["scene_label"]
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