#%%
from mmpose.apis import MMPoseInferencer
from datetime import datetime
import subprocess
import os
os.chdir('../..')

class MMPoseModelCoach:
    command = 'python'
    script = 'tools/train.py'
    detector_model = {  #rtmdet
        "det_model": 'mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py',
        "det_weights": 'checkpoints/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth',
    }

    def __init__(self, config, resume=True, notes=''):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.config = config
        self.config_path = f'mmpose-synthetic-tune/dataset-coco/custom-configs/{self.config}'
        self.work_dir = f'mmpose-synthetic-tune/models/_train-{current_time}-{notes}'
        self.resume = '--resume' if resume == True else ''

        self.args = [
            self.command,
            self.script,
            self.config_path,
            '--work-dir',
            self.work_dir,
            self.resume,
        ]

    def train(self):
        subprocess.run(self.args)

    def visualize_results(self, model_ckpt, vis_input, radius=3, thickness=1):
        poser_model = {
            "pose2d": f'{self.work_dir}/{self.config}',
            "pose2d_weights": f'{self.work_dir}/{model_ckpt}',
        }

        inferencer = MMPoseInferencer(**poser_model, **self.detector_model, device='cuda:0')

        input_path = vis_input
        output_path = f'{self.work_dir}/vis_results'

        result_generator = inferencer(
            input_path,
            radius=radius,
            thickness=thickness,
            vis_out_dir=output_path,
            draw_heatmap=True,
            det_cat_ids=5
        )

        results = [res for res in result_generator]


#%%
base_hrnet = MMPoseModelCoach(
    config='cow20kp-base-hrnet.py',
    notes='base-hrnet-similar-val-500-epochs'
)
base_hrnet.train()

#%%

base_hrnet.visualize_results(
    model_ckpt='epoch_500.pth',
    vis_input='mmpose-synthetic-tune/dataset-coco/data/test-footage/cow.png'
)

#%%
base_apk10k = MMPoseModelCoach(
    config='cow20kp-base-ap10k-hrnet.py',
    notes='base-ap10k'
)
base_apk10k.train()
#%%
base_apk10k.visualize_results(
    model_ckpt='epoch_210.pth',
    vis_input='mmpose-synthetic-tune/dataset-coco/data/test-footage/cow.png'
)
