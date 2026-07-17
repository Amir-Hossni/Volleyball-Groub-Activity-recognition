from Data.dataset import VolleyballDataset
from Data.extract_features import prepare_model
from pathlib import Path
import yaml
BASE_DIR = Path(__file__).resolve().parent

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
config = load_config("config.yaml")
data_root = Path(config["Data"]["DATA_ROOT"])


# ==========================
# Load Config
# ==========================

config = load_config()

data_cfg = config["Data"]

data_root = Path(data_cfg["DATA_ROOT"])

videos_path = data_root / data_cfg["PATHS"]["VIDEOS_PATH"]

annot_root = data_root / data_cfg["PATHS"]["TRACKING_ANNOTATION_PATH"]

scene_to_idx = data_cfg["CATEGORIES"]["SCENE_TO_IDX"]

train_ids = data_cfg["SPLIT"]["TRAIN_IDS"]


def main():

    # ==========================
    # Transform
    # ==========================

    transform = prepare_model(
        image_level=False
    )


    # ==========================
    # Dataset
    # ==========================

    train_dataset = VolleyballDataset(

        videos_path=videos_path,

        annot_root=annot_root,

        split_ids=train_ids,

        scene_to_idx=scene_to_idx,

        mode="person",

        transform=transform

    )


    # ==========================
    # Dataset Checks
    # ==========================

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Videos Path : {videos_path}")
    print(f"Annotation Path : {annot_root}")

    print(f"Dataset Size : {len(train_dataset)}")

    print()


    # ==========================
    # First Sample
    # ==========================

    sample = train_dataset[0]

    print("=" * 60)
    print("First Sample")
    print("=" * 60)

    for key, value in sample.items():

        if hasattr(value, "shape"):

            print(f"{key:<15}: {value.shape}")

        else:

            print(f"{key:<15}: {value}")



if __name__ == "__main__":
    main()