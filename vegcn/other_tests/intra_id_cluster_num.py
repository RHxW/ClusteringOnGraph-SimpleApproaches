import os
import numpy as np
import shutil

from vegcn.inference_gcnv import inference_gcnv
from vegcn.config.gcnv_config import CONFIG


def get_intra_id_cluster_num(cfg, id_dir, features_dir):
    """
    判断一个文件夹下图片能够聚成多少个类（用来判断是否为混乱id）
    :param cfg:
    :param id_dir: 里面是图片
    :param features_dir: 里面是特征文件，文件名与id_dir中的图片名一致
    :return:
    """
    imgs = os.listdir(id_dir)
    feats = []
    for i in range(len(imgs)):
        name = imgs[i].split(".")[0]
        feats.append(np.load(features_dir + name + ".npy"))
    feats = np.array(feats)
    feature_path = "./tmp_feature.npy"
    np.save(feature_path, feats)
    pred_labels, gcn_features = inference_gcnv(cfg, feature_path)
    shutil.rmtree(feature_path)
    n = len(set(pred_labels))
    return n