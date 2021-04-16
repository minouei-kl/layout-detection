_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
    ) )


dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[30.3344, 30.4473, 30.3461], std=[56.7147, 56.9299, 56.8265], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256,256), keep_ratio=True),
    # dict(type='MyTransform'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[30.3344, 30.4473, 30.3461],
        std=[56.7147, 56.9299, 56.8265],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256,256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='MyTransform'),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[30.3344, 30.4473, 30.3461],
                std=[56.7147, 56.9299, 56.8265],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        classes=('text', 'title', 'list', 'table', 'figure'),
        ann_file='/home/minouei/Downloads/datasets/publaynet/minival.json',
        img_prefix='/home/minouei/Downloads/datasets/publaynet/val/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        classes=('text', 'title', 'list', 'table', 'figure'),
        ann_file='/home/minouei/Downloads/datasets/publaynet/minival.json',
        img_prefix='/home/minouei/Downloads/datasets/publaynet/val/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=('text', 'title', 'list', 'table', 'figure'),
        ann_file='/home/minouei/Downloads/datasets/publaynet/val.json',
        img_prefix='/home/minouei/Downloads/datasets/publaynet/val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
lr_config = dict(_delete_=True,
    policy='cyclic',
    target_ratio=(10, 1),
    cyclic_times=12,
    step_ratio_up=0.5,
    # policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=0.001,
    # step=[8, 11]
)
total_epochs = 12
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
rpn_weight = 0.7
work_dir = './work_dirs/crpn_faster_rcnn_r50_fpn_1x_coco'
gpu_ids = range(0, 1)
