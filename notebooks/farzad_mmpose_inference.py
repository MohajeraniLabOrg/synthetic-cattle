#%%
import cv2
import mmcv
import numpy as np
from mmpose.apis import inference_topdown, init_model
import torch

config_file = '/home/galiold/projects/mmpose/mmpose-synthetic-tune/mmpose/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py'
checkpoint_file = '/home/galiold/projects/mmpose/mmpose-synthetic-tune/checkpoints/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth'

pose_model = init_model(config_file, checkpoint_file, device='cuda:0' if torch.cuda.is_available() else 'cpu')

image_path = "/home/galiold/projects/mmpose/mmpose-synthetic-tune/dataset-coco/data/sample-15/images/6_img022.png"
image = mmcv.imread(image_path)

try:
    pose_results = inference_topdown(
        pose_model,
        image
    )
except Exception as e:
    print(f"Error during pose estimation: {e}")
    cv2.imshow("Highlighted Cow Head", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

#%%
if pose_results:
    pose_result = pose_results[0]

    keypoint_labels = pose_model.cfg.data.test.get('keypoint_id_to_label', None)
    if keypoint_labels is None:
        print("Keypoint labels not found in the model config.")
        cv2.imshow("Highlighted Cow Head", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    head_landmarks = [
        pose_result['keypoints'][keypoint_labels['nose']],
        pose_result['keypoints'][keypoint_labels['left_eye']],
        pose_result['keypoints'][keypoint_labels['right_eye']],
    ]

    head_landmarks = [
        (int(lm[0]), int(lm[1]))
        for lm in head_landmarks
        if lm[2] > 0.5 and 0 <= lm[0] < image.shape[1] and 0 <= lm[1] < image.shape[0]
    ]

    if len(head_landmarks) >= 3:  

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(head_landmarks).astype(np.int32)], 255)

        highlighted_image = image.copy()
        highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
        highlighted_image[mask > 0] = [0, 255, 0]  # Green color
        highlighted_image = cv2.addWeighted(highlighted_image, 0.5, image, 0.5, 0)

        cv2.imshow("Highlighted Cow Head", highlighted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cv2.imshow("Highlighted Cow Head", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1 1]
# [1 1]
# [1 1]
# [1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1 1]
# [1 1]
# [1 1]
# [1 1 1 1]
# [1 1]
# [1 1]
# [1 1]
# [1]
# [1]
# [1]
# [1]
# [1 1]
# [1]
# [1]
# [1]
# [1 1]
# [1]
# [1]
# []
# [1]
# []
# [1]
# [1]
# [1 1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# [1]
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
# []
