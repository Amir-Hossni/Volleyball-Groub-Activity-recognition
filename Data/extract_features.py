# import numpy as np
# import cv2
# import torch

# from pathlib import Path
# import torch.nn as nn
# import torchvision.models as models
import torchvision.transforms as transforms
# from PIL import Image
# from .volleyball_annot_loader import load_tracking_annot







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

    # Check if a GPU is available and if not, use a CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # resnet50 alexnet
    # model = models.resnet50(pretrained=True)  # You can also use 'mobilenet_v3_large'

    # # Remove the classification head (i.e., the fully connected layers)
    # model = nn.Sequential(*(list(model.children())[:-1]))

    # # Send the model to the device (CPU or GPU)
    # model.to(device)

    # # Set the model to evaluation mode
    # model.eval()

    # return model, preprocess
    return preprocess


# def extract_features(clip_dir_path, annot_file, output_file, model, preprocess, image_level=False):
#     frame_boxes = load_tracking_annot(annot_file)

#     with torch.no_grad():
#         for frame_id, boxes_info in frame_boxes.items():
#             try:
#                 img_path = Path(clip_dir_path) / f"{frame_id}.jpg"
#                 image = Image.open(img_path).convert('RGB')

#                 if image_level:
#                     preprocessed_image = preprocess(image).unsqueeze(0)
#                     dnn_repr = model(preprocessed_image)
#                     dnn_repr = dnn_repr.view(1, -1)
#                 else:
#                     # for each image player's box, extract cropped images, extract features
#                     preprocessed_images = []
#                     for box_info in boxes_info:
#                         x1, y1, x2, y2 = box_info.box
#                         cropped_image = image.crop((x1, y1, x2, y2))

#                         if True:   # visualize a crop
#                             cv2.imshow('Cropped Image', np.array(cropped_image))
#                             cv2.waitKey(0)

#                         preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))

#                     preprocessed_images = torch.cat(preprocessed_images)
#                     dnn_repr = model(preprocessed_images)    # Batch Processing
#                     dnn_repr = dnn_repr.view(len(preprocessed_images), -1)  # 12 x 2048 for resnet 50

#                 # uncomment to save features
#                 #np.save(output_file, dnn_repr.numpy())
#             except Exception as e:
#                 print(f"An error occurred: {e}")



