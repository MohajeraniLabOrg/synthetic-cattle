# %% Functions
import numpy as np

def group_iter(iterable, n):
    return zip(*[iter(iterable)]*n)

def parse_keypoints(keypoints: list):
    parsed_keypoints = {}
    idx = 1
    Xs = Ys = []

    for (x, y, vis) in group_iter(keypoints, 3):
        parsed_keypoints[idx] = { "x": int(x), "y": int(y), "vis": vis }
        idx += 1
        if vis != 0:
            Xs.append(x)
            Ys.append(y)
        
    return parsed_keypoints, (np.mean(Xs), np.mean(Ys))
# %%
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


for a_idx, annot in enumerate(dataset['annotations']):
    kps = annot['keypoints']
    parsed_kps, centroid = parse_keypoints(kps)

    edited_kps = []

    for k_idx, kp in enumerate(parsed_kps.values()):
        x = kp['x']
        y = kp['y']

        if kp['vis'] != 2:
            (x, y) = centroid

        if k_idx == 6:
            y -= 90

        edited_kps.extend([x, y , kp['vis']])

    annot['keypoints'] = edited_kps

with open('./cvat-validation-modified.json' ,'w') as f:
    f.write(json.dumps(dataset))