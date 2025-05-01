point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesOfflineLocalMapDataset_v2'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=['divider', 'ped_crossing', 'boundary']),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='CustomNuScenesOfflineLocalMapDataset_v2',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_map_infos_temporal_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=['divider', 'ped_crossing', 'boundary']),
            dict(type='CustomCollect3D', keys=['img'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=True,
            segmap=True,
            seg_classes=1,
            segmap_classes=1,
            segmap_select_indexes=[1],
            feat_down_sample=16,
            pv_thickness=1,
            lidar_bev_maps=True,
            lidar_bev_maps_count=2,
            lidar_bev_maps_select_indexes=[0, 1]),
        use_valid_flag=True,
        bev_size=(200, 100),
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        queue_length=1),
    val=dict(
        type='CustomNuScenesOfflineLocalMapDataset_v2',
        ann_file='data/nuscenes/nuscenes_map_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        with_gt=False,
                        with_label=False,
                        class_names=['divider', 'ped_crossing', 'boundary']),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/nuscenes/',
        map_ann_file='data/nuscenes/nuscenes_map_anns_val.json',
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=True,
            segmap=True,
            seg_classes=1,
            segmap_classes=1,
            segmap_select_indexes=[1],
            feat_down_sample=16,
            pv_thickness=1,
            lidar_bev_maps=True,
            lidar_bev_maps_count=2,
            lidar_bev_maps_select_indexes=[0, 1]),
        bev_size=(200, 100),
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        samples_per_gpu=1),
    test=dict(
        type='CustomNuScenesOfflineLocalMapDataset_v2',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_map_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        with_gt=False,
                        with_label=False,
                        class_names=['divider', 'ped_crossing', 'boundary']),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        map_ann_file='data/nuscenes/nuscenes_map_anns_val.json',
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=True,
            segmap=True,
            seg_classes=1,
            segmap_classes=1,
            segmap_select_indexes=[1],
            feat_down_sample=16,
            pv_thickness=1,
            lidar_bev_maps=True,
            lidar_bev_maps_count=2,
            lidar_bev_maps_select_indexes=[0, 1]),
        bev_size=(200, 100),
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=['divider', 'ped_crossing', 'boundary']),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=2,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    with_gt=False,
                    with_label=False,
                    class_names=['divider', 'ped_crossing', 'boundary']),
                dict(type='CustomCollect3D', keys=['img'])
            ])
    ],
    metric='chamfer',
    save_best='NuscMap_chamfer/mAP',
    rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/segmap_24ep'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.15, 0.15, 4.0]
dbound = [1.0, 35.0, 0.5]
grid_config = dict(
    x=[-30.0, -30.0, 0.15],
    y=[-15.0, -15.0, 0.15],
    z=[-10, 10, 20],
    depth=[1.0, 35.0, 0.5])
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
map_classes = ['divider', 'ped_crossing', 'boundary']
num_vec = 50
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = 3
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 1
_num_points_in_pillar_ = 8
bev_h_ = 200
bev_w_ = 100
queue_length = 1
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    segmap=True,
    seg_classes=1,
    segmap_classes=1,
    segmap_select_indexes=[1],
    feat_down_sample=16,
    pv_thickness=1,
    lidar_bev_maps=True,
    lidar_bev_maps_count=2,
    lidar_bev_maps_select_indexes=[0, 1])
model = dict(
    type='MapTRv2',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='SegMapHead',
        bev_h=200,
        bev_w=100,
        num_query=900,
        num_vec_one2one=100,
        num_vec_one2many=600,
        k_one2many=6,
        num_pts_per_vec=20,
        num_pts_per_gt_vec=20,
        dir_interval=1,
        query_embed_type='instance',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=True,
            segmap=True,
            seg_classes=1,
            segmap_classes=1,
            segmap_select_indexes=[1],
            feat_down_sample=16,
            pv_thickness=1,
            lidar_bev_maps=True,
            lidar_bev_maps_count=2,
            lidar_bev_maps_select_indexes=[0, 1]),
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                num_points_in_pillar=8,
                return_intermediate=False,
                with_height_refine=True,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='HeightKernelAttention',
                            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                            num_points_in_pillar=8,
                            attention=dict(
                                type='MSDeformableAttentionKernel',
                                embed_dims=256,
                                num_heads=8,
                                dilation=1,
                                kernel_size=(2, 4),
                                num_levels=1),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                query_pos_embedding='instance',
                num_pts_per_vec=20,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='InstancePointAttention',
                            embed_dims=256,
                            num_levels=1,
                            num_pts_per_vec=20)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 4.0],
            num_classes=3),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=100),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_seg=dict(type='SimpleLoss', pos_weight=4.0, loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss', pos_weight=1.0, loss_weight=2.0),
        loss_segmap=dict(type='DiceLoss', loss_weight=2.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 4.0],
            point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            out_size_factor=4,
            assigner=dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]))))
samples_per_gpu = 1
optimizer = dict(
    type='AdamW',
    lr=0.0006,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    cumulative_iters=32, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=17600,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
fp16 = dict(loss_scale=512.0)
gpu_ids = range(0, 1)
