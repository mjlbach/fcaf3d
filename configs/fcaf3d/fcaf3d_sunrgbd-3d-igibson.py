_base_ = ['fcaf3d.py']
n_points = 100000

model = dict(
    neck_with_head=dict(
        n_classes=231,
        n_reg_outs=8))

dataset_type = 'SUNRGBDIGibsonDataset'
data_root = '/svl/u/alanlou/mmdetection3d/data/sunrgbd/'
# Copied from data/sunrgbd/category_list.txt.
class_names = (
    '7', '11', '14', '15', '16', '17', '18', '21', '22', '23',
    '25', '26', '27', '31', '32', '33', '34', '35', '36', '37',
    '38', '39', '40', '41', '42', '43', '46', '47', '48', '49',
    '50', '51', '53', '54', '56', '58', '59', '60', '61', '62',
    '63', '65', '66', '67', '69', '71', '72', '73', '76', '78',
    '79', '86', '87', '89', '91', '94', '96', '97', '99',
    '104', '105', '108', '109', '114', '119', '120', '121',
    '122', '126', '127', '128', '129', '131', '133', '135',
    '136', '137', '138', '140', '142', '143', '148', '150',
    '151', '152', '156', '158', '159', '160', '161', '176',
    '178', '179', '180', '181', '183', '184', '186', '187',
    '188', '189', '194', '195', '198', '199', '201', '202',
    '203', '207', '211', '212', '215', '216', '218', '222',
    '224', '225', '226', '227', '228', '229', '230', '231',
    '232', '234', '235', '236', '237', '238', '239', '241',
    '242', '244', '245', '247', '249', '251', '252', '253',
    '255', '258', '259', '260', '263', '264', '265', '268',
    '271', '272', '274', '275', '276', '277', '279', '280',
    '281', '283', '284', '285', '287', '289', '290', '291',
    '293', '294', '296', '297', '303', '305', '306', '307',
    '308', '309', '312', '313', '314', '315', '316', '317',
    '319', '320', '321', '322', '323', '324', '325', '326',
    '328', '330', '332', '334', '335', '336', '337', '338',
    '342', '343', '345', '346', '348', '349', '350', '351',
    '352', '353', '355', '356', '358', '361', '362', '365',
    '366', '368', '369', '370', '371', '372', '373', '376',
    '377', '378', '380', '382', '384', '387', '389', '392',
    '393', '394', '395', '396')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='IndoorPointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(480, 480),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='IndoorPointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_train.pkl',
        pipeline=train_pipeline,
        filter_empty_gt=True,
        classes=class_names,
        box_type_3d='Depth'),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_debug2.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_debug2.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
