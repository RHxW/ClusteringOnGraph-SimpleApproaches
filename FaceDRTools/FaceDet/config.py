cfg_train = {
    'name': 'Retinaface Training',
    'network': 'resnet50',
    'training_dataset': r'F:\Data\face_dect\widerface\train\label.txt',
    'num_workers': 1,
    'lr': 1e-3,
    'momentum': 0.9,
    'pretrain': False,
    # 'resume_path': 'weights/',
    'resume_epoch': 0,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'rgb_mean': (104, 117, 123),
    'num_classes': 2,
    'gpu_train': True,
    'batch_size': 1,
    'epoch': 100,
    'ckpt_root': './weights/',
    'log_root': './log/',

    'ngpu': 1
}


cfg_align = {
    'name': 'Retinaface',

    'det_model': r'../ckpts/detect/Resnet50_Final.pth',
    # 'det_model': r'../weights/E_73-Lb_0.0496-Ll_0.2914-Lc_0.3796.pth',
    'network': 'resnet50',

    'recg_model': r'../ckpts/recg/Backbone_IR_SE_50_Epoch_55_Batch_512864_Time_2020-08-25-00-10_checkpoint.pth',

    # 'infer_size': (320, 320),  # 传给检测网络的图像宽，高， 先右下角padding成方形，再缩放
    'infer_size': (-1, -1),      # -1 表示原图， 不做缩放

    'use_box_scale': True,      # if False， 对齐接收原图， 已经设置为True 4.28
    'box_scale': 1.5,  # 检测框外扩比例， 之前外扩之后传给对齐代码，现在对齐代码接收的是原图，所以这里无用~

    'aligned_size': (96, 112),    # 对齐图像目标 宽， 高
    'aligned_scale': 1.0,

    'rgb_mean': (104, 117, 123),    # 实际图像是BGR, 这里顺序也是BGR

    'confidence_threshold': 0.3,    # 和vis_thres 应该没区别~~两处判断
    'top_k': 500,
    'nms_threshold': 0.4,
    'keep_top_k': 10,
    'vis_thres': 0.3,   # 检测框置信度

    'cpu': False
}


cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'pretrain': True,
    'loc_weight': 2.0,

    'decay1': 70,
    'decay2': 90,
    'image_size': 840,

    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
