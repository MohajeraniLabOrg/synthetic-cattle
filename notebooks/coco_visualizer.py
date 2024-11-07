# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import random

def group_iter(iterable, n):
    return zip(*[iter(iterable)]*n)

def parse_keypoints(keypoints: list):
    parsed_keypoints = {}
    idx = 1
    for (x, y, vis) in group_iter(keypoints, 3):
        parsed_keypoints[idx] = { "x": int(x), "y": int(y), "vis": vis }
        idx += 1
    return parsed_keypoints

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import json
import cv2
from matplotlib import pyplot as plt
import numpy as np

with open('../dataset-coco/annotations/validation-cvat.json') as f:
    dataset = json.loads(f.read())


images = dataset['images']
annotations = dataset['annotations']
skeleton = dataset['categories'][0]['skeleton']
skeleton_lbls = dataset['categories'][0]['keypoints']
kp_colors = plt.cm.hsv(np.linspace(0, 1, 21)) * 255
link_colors = plt.cm.hsv(np.linspace(0, 1, 22)) * 255


for image, annot in zip(images, annotations):
    img_file = f"../dataset-coco/data/{image['file_name']}"
    print(img_file)

    img = cv2.imread(img_file)

    # Draw bbox
    [topx, topy, w, h] = [ int(x) for x in annot['bbox'] ]
    img = cv2.rectangle(img, (topx, topy), (topx + w, topy + h), color=(0, 255, 0), thickness=2)

    # Draw keypoints

    keypoints = parse_keypoints(annot['keypoints'])

    for idx, k in enumerate(keypoints.values()):
        if k['vis'] == 0:
            continue
        img = cv2.circle(img, (k['x'], k['y']), radius=3, color=kp_colors[idx], thickness=-1)
        img = cv2.putText(img, f'{idx}. {skeleton_lbls[idx]}', (k['x'] + 5, k['y']), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=.75, color=(0, 0, 255), thickness=1)


    # Draw links
    for idx, [k1, k2] in enumerate(skeleton):
        if keypoints[k1]['vis'] == 0 or keypoints[k2]['vis'] == 0:
            continue
        img = cv2.line(img, (keypoints[k1]['x'], keypoints[k1]['y']),
                       (keypoints[k2]['x'], keypoints[k2]['y']), color=link_colors[idx], thickness=1)

    cv2.imwrite(f"../dataset-coco/labeled-data/{image['file_name']}", img)

    # plt.figure(figsize=(13,13))
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # break

# %%
