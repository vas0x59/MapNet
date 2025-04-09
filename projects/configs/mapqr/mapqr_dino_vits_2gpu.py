_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-15.0, -30.0,-2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4.0]
dbound=[1.0, 35.0, 0.5]

grid_config = {
    'x': [-30.0, -30.0, 0.15], # useless
    'y': [-15.0, -15.0, 0.15], # useless
    'z': [-10, 10, 20],        # useless
    'depth': [1.0, 35.0, 0.5], # useful
}


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
num_vec=50
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
_num_points_in_pillar_ = 8
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    segmap=True,
    seg_classes=1,
    segmap_classes=1, # layers=['ped_crossing', 'drivable_area', 'road_segment']
    feat_down_sample=32,
    pv_thickness=1,
)
pretrained = 'ckpts/dino_vits16.pth'
model = dict(
    type='SegMapDino',
    use_grid_mask=True,
    video_test_mode=False,
    img_backbone=dict(
        type='ViTAdapter',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        use_extra_extractor=False,
        layer_scale=False,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,
                     14, 14, None, 14, 14, None],
        pretrained=pretrained),
    img_neck=dict(
        type='FPN',
        in_channels=[384], # , 384, 384, 384]
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        norm_cfg=dict(type='MMSyncBN', requires_grad=True)),
    pts_bbox_head=dict(
        type='SegMapHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=100,
        num_vec_one2many=600,
        k_one2many=6,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        # query_embed_type='instance_pts',
        query_embed_type='instance',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        # z_cfg=z_cfg,
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=_num_points_in_pillar_,
                return_intermediate=False,
                with_height_refine=True,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                           type='HeightKernelAttention',
                           pc_range=point_cloud_range,
                           num_points_in_pillar=_num_points_in_pillar_,
                           attention=dict(
                               type='MSDeformableAttentionKernel',
                               embed_dims=_dim_,
                               num_heads=_num_points_in_pillar_,
                               dilation=1,
                               kernel_size=(2, 4),
                               num_levels=_num_levels_),
                           embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                query_pos_embedding='instance',
                num_pts_per_vec=fixed_ptsnum_per_pred_line,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='InstancePointAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_pts_per_vec=fixed_ptsnum_per_pred_line,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_seg=dict(type='DiceLoss', 
            # pos_weight=4.0,
            loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss', 
                    pos_weight=1.0,
                    loss_weight=2.0),
        loss_segmap=dict(type='DiceLoss',
                    loss_weight=2.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesOfflineLocalMapDataset_v2'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(
    #     type='CustomLoadPointsFromFile', # LoadPointsFromFile
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     reduce_beams=32),
    # dict(type='CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    # dict(type='PadMultiViewImageDepth', size_divisor=32), 
    dict(type='PadMultiViewImage', size_divisor=32), 
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,class_names=map_classes),
    dict(type='CustomCollect3D', keys=['img' ]) # , 'gt_depth'
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
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
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]
samples_per_gpu=2
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4, # TODO 12
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_map_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        aux_seg=aux_seg_cfg,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline, 
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names, 
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(cumulative_iters=32, grad_clip=dict(max_norm=35, norm_type=2)) # GradientCumulativeFp16OptimizerHook
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='NuscMap_chamfer/mAP', rule='greater')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='MapNet',   # Название проекта в WandB
                name='dino_vits16',     # Имя эксперимента
                config=dict(                # Дополнительные настройки эксперимента
                    batch_size=samples_per_gpu*2,
                    model='mapqr',
                )
            )
        )
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
# find_unused_parameters=True