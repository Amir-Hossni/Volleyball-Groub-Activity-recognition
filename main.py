
from pathlib import Path
import pickle

import yaml
from torch.utils.data import DataLoader
import torch

from utlis.early_stopping import EarlyStopping
from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model

# from Baseline2.model_B2 import B2Model
from Baseline3.model_B3 import PersonClassifierB3, GroupClassifierB3

# from Baseline2.training_B2 import train
from Baseline3.training_person_B3 import train_person_B3

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


# Dataset
train_dataset = VolleyballDataset(
    videos_path=videos_path,
    pkl_path=pkl_path,
    split_ids=train_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person",
    transform=transform
)

val_dataset = VolleyballDataset(
    videos_path=videos_path,
    pkl_path=pkl_path,
    split_ids=val_ids,
    scene_to_idx=scene_to_idx,
    player_to_idx=player_to_idx,
    mode="person",
    transform=transform
)


# DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
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

#B2
# model = B2Model(
#     num_players=12,
#     num_classes=len(scene_to_idx),
#     pretrained=True
# )

# B3
model = PersonClassifierB3(
    num_classes=len(player_to_idx),
    pretrained=True
)
# person_model.load_state_dict(torch.load("person.pth"))
# backbone = person_model.backbone
# group_model = GroupClassifierB3(backbone)

if torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = torch.nn.DataParallel(model)

model = model.to(device)


# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

early_stopping = EarlyStopping(
    patience=10,
    mode="max"
)

if __name__ == "__main__":
    
    
    # train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     device,
    #     epochs=50,
    #     save_path="/kaggle/working/best_B2_model.pth"
    # )
    
    
#     train_person_B3(
#     model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     device,
#     epochs=50,
#     save_path="/kaggle/working/best_B3_person_model.pth"
# )
    
    # create_pkl_version(videos_root=videos_path,annot_root=annot_root,save_path= "/kaggle/working/annot_all.pkl")
    
    import pickle
    from pathlib import Path


    # ==========================
    # Change this path
    # ==========================

    pkl_path = "/kaggle/input/datasets/amirhossni/annot-dataset/annot_all.pkl"


    # ==========================
    # Load PKL
    # ==========================

    with open(pkl_path, "rb") as f:
        annotations = pickle.load(f)


    print("=" * 50)
    print("PKL TYPE")
    print("=" * 50)

    print(type(annotations))


    print("\n" + "=" * 50)
    print("VIDEO KEYS")
    print("=" * 50)

    print(list(annotations.keys())[:5])


    # choose first video
    video_id = list(annotations.keys())[0]

    print("\nSelected video:")
    print(video_id)


    video = annotations[video_id]


    print("\n" + "=" * 50)
    print("VIDEO STRUCTURE")
    print("=" * 50)

    print(type(video))
    print(video.keys())


    # choose first clip

    clip_id = list(video.keys())[0]

    print("\nSelected clip:")
    print(clip_id)


    clip = video[clip_id]


    print("\n" + "=" * 50)
    print("CLIP STRUCTURE")
    print("=" * 50)

    print(type(clip))
    print(clip.keys())


    # frame boxes

    frame_boxes = clip["frame_boxes_dct"]


    print("\n" + "=" * 50)
    print("FRAME BOXES")
    print("=" * 50)

    print(type(frame_boxes))

    frame_id = list(frame_boxes.keys())[0]

    print("Frame ID:")
    print(frame_id)

    boxes = frame_boxes[frame_id]


    print("Boxes type:")
    print(type(boxes))

    print("Number of boxes:")
    print(len(boxes))


    # first box

    box = boxes[0]


    print("\n" + "=" * 50)
    print("BOX INFORMATION")
    print("=" * 50)


    print("Box type:")
    print(type(box))


    print("\nBox content:")
    print(box)


    print("\nBox keys (if dict):")

    if isinstance(box, dict):
        print(box.keys())


    print("\nBox attributes:")

    if hasattr(box, "__dict__"):
        print(box.__dict__)
    else:
        print("No __dict__")


    # ==========================
    # Path check
    # ==========================

    print("\n" + "=" * 50)
    print("IMAGE PATH CHECK")
    print("=" * 50)


    videos_path = Path(
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos"
    )


    image_path = (
        videos_path
        /
        str(video_id)
        /
        str(clip_id)
        /
        f"{frame_id}.jpg"
    )


    print(image_path)

    print("Exists:")
    print(image_path.exists())



    # ==========================
    # Check categories
    # ==========================

    print("\n" + "=" * 50)
    print("CATEGORIES")
    print("=" * 50)


    print("Clip category:")
    print(clip["category"])


    if isinstance(box, dict):
        print("Player category:")
        print(box["category"])