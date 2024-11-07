#%%
from mmdet.apis import DetInferencer
# %%
det_inferencer_options = dict(
    model       = '../../mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py',
    weights     = '../../checkpoints/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth',
    device      = 'cuda:0'
    )

inferencer = DetInferencer(**det_inferencer_options)
# %%
# Perform inference
res = inferencer(
    '../../data/cow.png',
    out_dir='../../vis_results/mmdet/',
)
#%%

maxscore = max(res['predictions'][0]['scores'])

# import json
# json.dumps(res)

labels = res['predictions'][0]['labels']
scores = res['predictions'][0]['scores']
bboxes = res['predictions'][0]['bboxes']

print(labels)

max_score = max(scores)

idx = 0
for label, score, bbox in zip(labels, scores, bboxes):
    if score > .8:
        print(f'label: {label}, score: {score}, bbox: {bbox}')

# %%

labels
# %%
masks