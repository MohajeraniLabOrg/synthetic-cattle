# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pycocotools import mask as mask_util
import hasty

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def polygonFromMask(mask): 
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []

    for obj in contours:
        coords = []
            
        for point in obj:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        polygons.append(coords)

    return polygons

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "../../checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json

with open('../dataset-coco/annotations/validation-feedlot.json') as file:
    dataset = json.loads(file.read())
    images = dataset['images']
    annotations = dataset['annotations']

for image, ann in zip(images[50:], annotations[50:]):
    file_name = image['file_name']
    print(file_name)
    img = cv2.imread(f"../dataset-coco/data/validation/{file_name}")
    bbox = np.array(ann['bbox'])

    predictor.set_image(img)

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=False,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        show_mask(mask, plt.gca())
        show_box(bbox, plt.gca())
        plt.title(f"{file_name} Mask {i+1} of {len(masks)}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

        # rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        # rle['counts'] = str(rle['counts'], encoding='utf-8')
        rle = hasty.label_utils.rle_encoding(mask.astype(np.uint8))
        print(rle)
        # plt.savefig(f"../../vis_results/sam/sam-{file_name}")
        # plt.close()

    break
# %%
import hasty