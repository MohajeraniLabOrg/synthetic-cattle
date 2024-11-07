batch_size = 16
sample_name = 'real-simillar'
data_batch = 3

_base_ = ['../mmpose/configs/_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=100, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        milestones=[60, 90],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='checkpoints/hrnet_w48-8ef0771d.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=20,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='checkpoints/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth'),
)

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/workspace/mmpose-synthetic-tune/datasets/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'annotations/20kp-{sample_name}-train-{data_batch}.json',
        data_prefix=dict(img=f'20kp-{sample_name}-data/'),
        metainfo=dict(from_file='custom-configs/20kp.py'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=batch_size//2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'annotations/20kp-{sample_name}-val-{data_batch}.json',
        data_prefix=dict(img=f'20kp-{sample_name}-data/'),
        metainfo=dict(from_file='custom-configs/20kp.py'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = dict(
    batch_size=batch_size//2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'annotations/20kp-test.json',
        data_prefix=dict(img=f'20kp-test-data/'),
        metainfo=dict(from_file='custom-configs/20kp.py'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + f'annotations/20kp-{sample_name}-val-{data_batch}.json')
test_evaluator = [
    dict(
        ann_file=data_root + f'annotations/20kp-test.json',
        format_only=False,
        outfile_prefix='test_annotations-real_simillar',
        type='CocoMetric'),
    dict(out_file_path='combined.pkl', type='DumpResults'),
]