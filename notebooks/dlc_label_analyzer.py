#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import cv2
import os
# import numpy as np
# from pycocotools import mask as mask_util
# from skimage import measure
# from shapely.geometry import Polygon, MultiPolygon


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reading DLC labels

folder = '/workspace/mmpose/data/real-dataset/7'

try:
    os.makedirs(f'{folder}/labeled')
except:
    pass

labels_csv = pd.read_csv(f'{folder}/labels.csv')
labels_csv.fillna(-1, inplace=True)

labels = labels_csv.iloc[:, 2:]
labels

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Defining functions


def pairwise(iterable):
    return zip(*[iter(iterable)]*2)


# def calculate_segmentations(mask): 
#     contours = measure.find_contours(mask, 0.5, positive_orientation='low')

#     segmentations = []
#     polygons = []

#     for contour in contours:
#         # Flip from (row, col) representation to (x, y)
#         # and subtract the padding pixel
#         for i in range(len(contour)):
#             row, col = contour[i]
#             contour[i] = (col - 1, row - 1)

#         # Make a polygon and simplify it
#         poly = Polygon(contour)
#         poly = poly.simplify(1.0, preserve_topology=False)
#         polygons.append(poly)
#         segmentation = np.array(poly.exterior.coords).ravel().tolist()
#         segmentations.append(segmentation)

#     return segmentations

# %% Visualizing labels on images with names

for r in labels.iterrows():
    entry = r[1]
    
    img_name: str = entry['img']
    keypoints = entry.iloc[1:].astype(int)

    img = cv2.imread(f'{folder}/{img_name}')

    kp_idx = 0
    for (name, x), (_, y) in pairwise(keypoints.items()):
        img = cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        img = cv2.putText(img, f'{kp_idx}. {name}', (x + 5, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=.75, color=(0, 0, 255), thickness=1)
        kp_idx += 1

    out_path = f"{folder}/labeled/{img_name.replace('.png', '-labeled.png')}"
    cv2.imwrite(out_path, img)
    # break



# %% Initializing detection model

from mmdet.apis import DetInferencer

det_inferencer_options = dict(
    model       = '../../mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py',
    weights     = '../../checkpoints/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth',
    device      = 'cuda:0'
    )

mmdet_inferencer = DetInferencer(**det_inferencer_options)

# %% Initializing segmentation model

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "../../checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)

# %% Initializing dataset

import json

try:
    with open('../dataset-coco/annotations/train-feedlot-real.json') as f:
        dataset = json.loads(f.read())
except FileNotFoundError:
    dataset = {
    "info": {
        "description": "Feedlot Cattle Dataset",
        "url": "",
        "version": "1.0",
        "year": 2022,
        "contributor": "CCBN",
        "date_created": "2024/03/15"
    },
    "licenses": [
        {
            "id": 1,
            "name": "The MIT License",
            "url": "https://www.mit.edu/~amini/LICENSE.md"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "cow",
            "supercategory": "",
            "keypoints": [
                "Back1",
                "Back2",
                "Back3",
                "Back4",
                "Head",
                "Nose",
                "Neck",
                "L_Shoulder",
                "L_Elbow",
                "L_F_Paw",
                "R_Shoulder",
                "R_Elbow",
                "R_F_Paw",
                "Belly",
                "L_Hip",
                "L_Knee",
                "L_H_Paw",
                "R_Hip",
                "R_Knee",
                "R_H_Paw"
            ],
            "skeleton": [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 7],
                [7, 6],
                [6, 5],
                [7, 8],
                [8, 9],
                [9, 10],
                [7, 11],
                [11, 12],
                [12, 13],
                [8, 14],
                [11, 14],
                [14, 15],
                [15, 16],
                [16, 17],
                [14, 18],
                [18, 19],
                [19, 20],
                [15, 1],
                [18, 1]
            ]
        }
    ],
    "images": [],
    "annotations": [],
}

# %% Adding data to dataset

images = []
annotations = []

try:
    last_image_id = dataset['images'][-1]['id']
except:
    last_image_id = 0

for r in labels.iterrows():
    last_image_id += 1

    entry = r[1]
    
    img_name: str = '7_' + entry['img']

    keypoints = entry.iloc[1:].astype(int)

    print(img_name)
    if os.path.isfile(f'{folder}/{img_name}'):
        img = cv2.imread(f'{folder}/{img_name}')
    else:
        print('Not Found')
        continue

    # Adding image info
    (h, w, _) = img.shape
    image_info = {
        'license': 1,
        "id": last_image_id,
        "file_name": img_name,
        "width": w,
        "height": h,
        "background": 1
    }

    # Adding annotation info
    annotation_info = {
        "id": last_image_id,
        "image_id": last_image_id,
        "category_id": 1,
        "bbox": [],
        'segmentation': [],
        "area": 0,
        "iscrowd": 0,
        "num_keypoints": 20,
        "keypoints": []
    }

    # Adding bbox info
    det_res = mmdet_inferencer(f'{folder}/{img_name}')
    det_labels = det_res['predictions'][0]['labels']
    det_scores = det_res['predictions'][0]['scores']
    det_bboxes = det_res['predictions'][0]['bboxes']

    max_score = max(det_scores)
    for label, score, bbox in zip(det_labels, det_scores, det_bboxes):
        if score == max_score:
            annotation_info["bbox"] = bbox
            annotation_info["area"] = bbox[2] * bbox[3]


    # Adding segmentation
    # sam_predictor.set_image(img)

    # masks, scores, logits = sam_predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=np.array(annotation_info["bbox"]),
    #     multimask_output=False,
    # )

    # coco_mask = polygonFromMask(np.array(masks[0]).astype(np.uint8))
    # coco_mask = calculate_segmentations(masks[0].astype(np.uint8))
    # annotation_info['segmentation'] = []

    # Adding keypoints
    annotation_kps = []
    for idx, ((_, x), (_, y)) in enumerate(pairwise(keypoints.items())):
        if idx == 20:
            break

        if x != -1 and y != -1:
            annotation_kps.extend([x, y, 2])
        else:
            annotation_kps.extend([10, 10, 1])

    annotation_info['keypoints'] = annotation_kps

    images.append(image_info)
    annotations.append(annotation_info)

dataset['images'].extend(images)
dataset['annotations'].extend(annotations)
    # break
    
# %%
with open('../dataset-coco/annotations/train-feedlot-real.json', 'w') as f:
    f.write(json.dumps(dataset))
# %%
# dataset
# %%
