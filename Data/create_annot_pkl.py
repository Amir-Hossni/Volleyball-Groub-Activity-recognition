from .volleyball_annot_loader import load_volleyball_dataset

import pickle

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
if __name__ == "__main__":
    
    
    import pickle

    with open("annot_all.pkl", "rb") as f:
        annot = pickle.load(f)

    print(type(annot))

    video_id = next(iter(annot))
    print("Video ID:", video_id)

    clip_id = next(iter(annot[video_id]))
    print("Clip ID:", clip_id)

    print("Clip keys:", annot[video_id][clip_id].keys())

    frame_id = next(iter(annot[video_id][clip_id]["frame_boxes_dct"]))
    print("Frame ID:", frame_id)

    boxes = annot[video_id][clip_id]["frame_boxes_dct"][frame_id]

    print(type(boxes))
    print(len(boxes))

    box = boxes[0]

    print(type(box))
    print(vars(box))