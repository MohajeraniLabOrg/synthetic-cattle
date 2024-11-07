#%%
from mmpose.apis import MMPoseInferencer
# import matplotlib.pyplot as plt

# Det models
rtmdet = dict(
    det_model       = '../../mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py',
    det_weights     = '../../checkpoints/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth',
)
# Pose models
ap10k_path = '../../mmpose-synthetic-tune/models/pretrained-hrnet_w48_ap10k_256x256-d95ab412_20211029'
ap10k = dict(
    pose2d          = f'{ap10k_path}/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py',
    pose2d_weights  = f'{ap10k_path}/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth',
)

hrnet_path = '../../mmpose-synthetic-tune/models/pretrained-hrnet_w48-8ef0771d'
hrnet = dict(
    pose2d          = f'{hrnet_path}/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py',
    pose2d_weights  = f'{hrnet_path}/hrnet_w48-8ef0771d.pth'
)
cow20k_path = '../../mmpose-synthetic-tune/models/_train-2024-04-15_10-51-49-base-ap10k'
cow20k = dict(
    pose2d          = f'{cow20k_path}/td-hm_hrnet-w48_8xb64-210e_20kp-256x256.py',
    pose2d_weights  = f'{cow20k_path}/last_checkpoint',
)

detector_model      = rtmdet
poser_model         = cow20k
model_path          = cow20k_path
    
inferencer = MMPoseInferencer(**poser_model, **detector_model, device='cuda:0')

#%%
# file = 'cow.png'
# input_path = f'../../data/{file}'
input_path = '../../mmpose-synthetic-tune/dataset-coco/data/20kp/images/train'
output_path = f'{model_path}/vis_results'

result_generator = inferencer(
    input_path,
    radius=3,
    thickness=1,
    vis_out_dir=output_path,
    draw_heatmap=True,
    det_cat_ids=5
)

results = [res for res in result_generator]

#%%

img = plt.imread(f'{output_path}/{file}')
pose = img[:img.shape[0]//2, :, :]
heatmap = img[img.shape[0]//2:, :, :]

fig = plt.figure(figsize=(15, 15))
ax_array = fig.subplots(2, 1)
ax_array[0].imshow(pose)
ax_array[1].imshow(heatmap)
# %%
results[0]['predictions'][0][0].keys()