
import torchvision.transforms as transforms



def prepare_model(image_level = False):
    if image_level:
        # image has a lot of space around objects. Let's crop around
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # already croped box. Don't crop more
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    return preprocess






# # ==========================
# # Change this path
# # ==========================

# pkl_path = "/kaggle/input/datasets/amirhossni/annot-dataset/annot_all.pkl"


# # ==========================
# # Load PKL
# # ==========================

# with open(pkl_path, "rb") as f:
#     annotations = pickle.load(f)


# print("=" * 50)
# print("PKL TYPE")
# print("=" * 50)

# print(type(annotations))


# print("\n" + "=" * 50)
# print("VIDEO KEYS")
# print("=" * 50)

# print(list(annotations.keys())[:5])


# # choose first video
# video_id = list(annotations.keys())[0]

# print("\nSelected video:")
# print(video_id)


# video = annotations[video_id]


# print("\n" + "=" * 50)
# print("VIDEO STRUCTURE")
# print("=" * 50)

# print(type(video))
# print(video.keys())


# # choose first clip

# clip_id = list(video.keys())[0]

# print("\nSelected clip:")
# print(clip_id)


# clip = video[clip_id]


# print("\n" + "=" * 50)
# print("CLIP STRUCTURE")
# print("=" * 50)

# print(type(clip))
# print(clip.keys())


# # frame boxes

# frame_boxes = clip["frame_boxes_dct"]


# print("\n" + "=" * 50)
# print("FRAME BOXES")
# print("=" * 50)

# print(type(frame_boxes))

# frame_id = list(frame_boxes.keys())[0]

# print("Frame ID:")
# print(frame_id)

# boxes = frame_boxes[frame_id]


# print("Boxes type:")
# print(type(boxes))

# print("Number of boxes:")
# print(len(boxes))


# # first box

# box = boxes[0]


# print("\n" + "=" * 50)
# print("BOX INFORMATION")
# print("=" * 50)


# print("Box type:")
# print(type(box))


# print("\nBox content:")
# print(box)


# print("\nBox keys (if dict):")

# if isinstance(box, dict):
#     print(box.keys())


# print("\nBox attributes:")

# if hasattr(box, "__dict__"):
#     print(box.__dict__)
# else:
#     print("No __dict__")


# # ==========================
# # Path check
# # ==========================

# print("\n" + "=" * 50)
# print("IMAGE PATH CHECK")
# print("=" * 50)


# videos_path = Path(
#     "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos"
# )


# image_path = (
#     videos_path
#     /
#     str(video_id)
#     /
#     str(clip_id)
#     /
#     f"{frame_id}.jpg"
# )


# print(image_path)

# print("Exists:")
# print(image_path.exists())



# # ==========================
# # Check categories
# # ==========================

# print("\n" + "=" * 50)
# print("CATEGORIES")
# print("=" * 50)


# print("Clip category:")
# print(clip["category"])


# if isinstance(box, dict):
#     print("Player category:")
#     print(box["category"])


#################################################################




# ==================================================
# VIDEO KEYS
# ==================================================
# ['0', '1', '10', '11', '12']

# Selected video:
# 0

# ==================================================
# VIDEO STRUCTURE
# ==================================================
# <class 'dict'>
# dict_keys(['13286', '13336', '13361', '13406', '13456', '18706', '18756', '18766', '18816', '18861', '18911', '18931', '18981', '19001', '24380', '24395', '24425', '24480', '24500', '24555', '24610', '24625', '24645', '25809', '25879', '25924', '25944', '25994', '26004', '26044', '26104', '30150', '34770', '35374', '35409', '35419', '35439', '35479', '35494', '35549', '35589', '35614', '35634', '35669', '35739', '35779', '35809', '35824', '35925', '3596', '3646', '3656', '3686', '36915', '3736', '3786', '38640', '42390', '43440', '44230', '44280', '44295', '44370', '44430', '44450', '44490', '44510', '44535', '45210', '45885', '48075', '52335', '53476', '53531', '53551', '53616', '53656', '53686', '53736', '53776', '53796', '53856', '53886', '54580', '54610', '54640', '54655', '54710', '55384', '55424', '55439', '55454', '55524', '55549', '55594', '55639', '55664', '58376', '58411', '58456', '58516', '58546', '58566', '58601', '58646', '58676', '58721', '59590', '59635', '59675', '59710', '59765', '59790', '59805', '60659', '60689', '60719', '60734', '60844', '60864', '60924', '61788', '61913', '61948', '61983', '62013', '62018', '62038', '62088', '62118', '72167', '7917', '7967', '8007', '8027', '8097', '8127', '8157', '8177', '8217', '8237', '8287', '8312', '8317', '8332', '9116', '9156', '9191', '9201', '9231', '9291', '9326', '9361'])

# Selected clip:
# 13286

# ==================================================
# CLIP STRUCTURE
# ==================================================
# <class 'dict'>
# dict_keys(['category', 'frame_boxes_dct'])

# ==================================================
# FRAME BOXES
# ==================================================
# <class 'dict'>
# Frame ID:
# 13281
# Boxes type:
# <class 'list'>
# Number of boxes:
# 12

# ==================================================
# BOX INFORMATION
# ==================================================
# Box type:
# <class 'dict'>

# Box content:
# {'category': 'standing', 'player_ID': 0, 'box': (42, 389, 104, 514), 'frame_ID': 13281, 'lost': 0, 'grouping': 0, 'generated': 1}

# Box keys (if dict):
# dict_keys(['category', 'player_ID', 'box', 'frame_ID', 'lost', 'grouping', 'generated'])

# Box attributes:
# No __dict__

# ==================================================
# IMAGE PATH CHECK
# ==================================================
# /kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos/0/13286/13281.jpg
# Exists:
# True

# ==================================================
# CATEGORIES
# ==================================================
# Clip category:
# r_set
# Player category:
# standing