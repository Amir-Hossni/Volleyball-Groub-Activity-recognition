from .volleyball_annot_loader import load_volleyball_dataset

import pickle
from pathlib import Path

def create_pkl_version(videos_root, annot_root, save_path):

    videos_annot = load_volleyball_dataset(
        videos_root,
        annot_root
    )

    with open(save_path, "wb") as file:
        pickle.dump(videos_annot, file)

    print(f"Saved: {save_path}")


# def test_pkl_version():
#     with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
#         videos_annot = pickle.load(file)

#     boxes: List[BoxInfo] = videos_annot['0']['13456']['frame_boxes_dct'][13454]
#     print(boxes[0].category)
#     print(boxes[0].box)
