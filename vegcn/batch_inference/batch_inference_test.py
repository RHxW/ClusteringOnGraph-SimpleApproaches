import numpy as np
import os

from vegcn.batch_inference.inf_gcnv_batch import GCNVBatchFeeder, GCNVInferenceBatch
from vegcn.config.gcnv_config import CONFIG
from utils import intdict2ndarray, read_meta

from evaluation.Purity_Diverse_V import get_DPV_measure
from evaluation.metrics import pairwise

def batch_inference(cfg, batch_num, feature_path):
    batch_feeder = GCNVBatchFeeder(cfg, batch_num, feature_path)
    N = batch_feeder.inst_num
    print("N: ",N)
    batch_infer = GCNVInferenceBatch(cfg, N)
    for i in range(batch_num):
        features_all, knn_all, idx_all, n = batch_feeder[i]
        print("batch: %d, n: %d, total_num: %d" % (i, n, len(features_all)))
        if features_all is None:
            break
        batch_infer.inference(features_all, knn_all, idx_all, n)

    pred_labels, gcn_features = batch_infer.get_cluster_result()
    return pred_labels, gcn_features



if __name__ == "__main__":
    data_root = "/tmp/pycharm_project_444/data/inf_data/lucheng/"
    feature_path = data_root + "feature.npy"
    cfg = CONFIG
    batch_num = 20
    pred_labels, gcn_features = batch_inference(cfg, batch_num, feature_path)
    print(gcn_features)
    save_path = data_root + "res/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    feature_save_path = save_path + "gcnv_inf_res.npy"
    label_save_path = save_path + "pred_label.txt"
    np.save(feature_save_path, gcn_features)

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
        print("pairwise F-score: avg_pre: %.6f, avg_rec: %.6f, fscore: %.6f" % (avg_pre, avg_rec, fscore))
        # V-measure
        diverse_score, purity_score, V_measure = get_DPV_measure(label_true, pred_labels)
        h, c, v = V_measure
        print("V-measure score: h: %.6f, c: %.6f, v: %.6f" % (h, c, v))
        print("*" * 50)