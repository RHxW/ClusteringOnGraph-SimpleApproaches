import os
import torch
import numpy as np

from vegcn.models.gcn_v import GCN_V
from vegcn.confidence import confidence_to_peaks
from vegcn.deduce import peaks_to_labels

from utils import (sparse_mx_to_torch_sparse_tensor, list2dict, read_meta,
                   intdict2ndarray, knns2ordered_nbrs, Timer, read_probs, l2norm, fast_knns2spmat,
                   build_symmetric_adj, row_normalize)
from utils.get_knn import build_knns
from vegcn.config.gcnv_config import CONFIG

from evaluation.Purity_Diverse_V import get_DPV_measure
from evaluation.metrics import pairwise


def inference_gcnv(cfg, feature_path):
    """
    不经过网络直接从原始特征构建的knn计算结果（pred_confs使用随机数）
    :param cfg:
    :param feature_path:
    :return:
    """
    torch.set_grad_enabled(False)
    k = cfg['knn']
    knn_method = cfg["knn_method"]
    is_norm_feat = cfg["is_norm_feat"]
    max_conn = cfg["max_conn"]
    tau_gcn = cfg["tau"]

    with Timer('read feature'):
        if not os.path.exists(feature_path):
            raise RuntimeError("feat_path not exists!!!")
        features = np.load(feature_path).astype(np.float32)
        if is_norm_feat:
            features = l2norm(features)
        inst_num = features.shape[0]

    # features = np.ascontiguousarray(features[:, :128])
    # use PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=128)
    # pca.fit(features)
    # features = pca.transform(features)

    with Timer('build knn graph'):
        knns = build_knns(features, knn_method, k)  # shape=(n, 2, k) NEW

    pred_confs = np.random.rand(inst_num)

    dists, nbrs = knns2ordered_nbrs(knns)
    # pred_dist2peak, pred_peaks = confidence_to_peaks(dists, nbrs, pred_confs, max_conn)
    num, _ = dists.shape
    pred_dist2peak = {i: [] for i in range(num)}
    pred_peaks = {i: [] for i in range(num)}
    for i, nbr in enumerate(nbrs):
        pred_dist2peak[i] = list(dists[i])
        pred_peaks[i] = list(nbr)


    pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, 0.6, inst_num)
    return pred_labels

if __name__ == "__main__":
    cfg = CONFIG
    data_root = "/tmp/pycharm_project_444/data/inf_data/idnum_1500/"
    feature_path = data_root + "feature.npy"
    pred_labels = inference_gcnv(cfg, feature_path)
    save_path = data_root + "res/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    label_save_path = save_path + "pred_label.txt"

    pl_lns = []
    for _lbl in pred_labels:
        pl_lns.append("%d\n" % int(_lbl))
    with open(label_save_path, "w") as f:
        f.writelines(list(pl_lns))

    # evaluation
    label_path = data_root + "label.txt"
    if os.path.exists(label_path):
        # label_path = data_root + "label.txt"
        lb2idxs, idx2lb = read_meta(label_path)
        inst_num = len(idx2lb)  # 样本数量
        label_true = intdict2ndarray(idx2lb)  # 真实标签

        avg_pre, avg_rec, fscore = pairwise(label_true, pred_labels)
        print("result class count: %d." % len(set(pred_labels)))
        print("pairwise F-score: avg_pre: %.6f, avg_rec: %.6f, fscore: %.6f" % (avg_pre, avg_rec, fscore))
        # V-measure
        # diverse_score, purity_score, V_measure = get_DPV_measure(label_true, pred_labels)
        # h, c, v = V_measure
        # print("V-measure score: h: %.6f, c: %.6f, v: %.6f" % (h, c, v))
        # print("*" * 50)