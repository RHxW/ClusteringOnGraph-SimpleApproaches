import os
import torch
import numpy as np

from vegcn.models.gcn_v import GCN_V
from vegcn.confidence import confidence_to_peaks
from vegcn.deduce import peaks_to_labels

from utils import (sparse_mx_to_torch_sparse_tensor, list2dict,
                   intdict2ndarray, knns2ordered_nbrs, Timer, read_probs, l2norm, fast_knns2spmat,
                   build_symmetric_adj, row_normalize)
from utils.get_knn import build_knns


def get_batch_idxs(n, batch_num, min_num_per_batch=10):
    """
    对idx按batch数量平均划分
    :param n:
    :param batch_num:
    :param min_num_per_batch:
    :return:
    """
    if batch_num * min_num_per_batch > batch_num:
        raise RuntimeError("batch_num too big")
    idx_all = list(range(n))
    num_per_batch = int(n / batch_num + 0.5)
    batch_idxs = []
    for i in range(batch_num):
        batch_idxs.append(idx_all[i * num_per_batch:(i + 1) * num_per_batch])
    return batch_idxs


class GCNVBatchFeeder():
    def __init__(self, cfg, batch_num, feature_path):
        self.k = cfg['knn']
        self.knn_method = cfg["knn_method"]
        self.feature_dim = cfg["feature_dim"]
        self.is_norm_feat = cfg["is_norm_feat"]

        with Timer('read feature'):
            if not os.path.exists(feature_path):
                raise RuntimeError("feat_path not exists!!!")
            self.features = read_probs(feature_path, self.inst_num, self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            self.inst_num = self.features.shape[0]

        self.batch_num = batch_num
        self.batch_idxs = get_batch_idxs(self.inst_num, self.batch_num)

        with Timer('build knn graph'):
            self.knns = build_knns(self.features, self.knn_method, self.k)  # shape=(n, 2, k) NEW

    def __getitem__(self, b):
        idx_focus = self.batch_idxs[b]
        n = len(idx_focus)
        knn_focus = [self.knns[i] for i in idx_focus]  # [N, 2, k]
        idx_others = set()  # no duplicate idx
        for knn_ in knn_focus:
            idx_others.update(set(knn_[0]))
        idx_all = idx_focus + list(idx_others)
        knn_all = [self.knns[i] for i in idx_all]
        features_all = self.features[idx_all]
        return features_all, knn_all, idx_all, n

    def __len__(self):
        return self.batch_num


class GCNVInferenceBatch():
    def __init__(self, cfg, N):
        torch.set_grad_enabled(False)
        self.device = cfg["device"]

        self.k = cfg['knn']
        self.knn_method = cfg["knn_method"]

        self.cut_edge_sim_th = cfg["cut_edge_sim_th"]
        self.max_conn = cfg["max_conn"]
        self.tau_gcn = cfg["tau"]

        # model
        self.feature_dim = cfg["feature_dim"]
        self.nhid = cfg["nhid"]
        self.nclass = cfg["nclass"]
        self.dropout = cfg["dropout"]
        self.model = GCN_V(self.feature_dim, self.nhid, self.nclass, self.dropout).to(self.device)
        print("Model: ", self.model)
        # load checkpoint
        checkpoint_path = cfg["checkpoint_path"]
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model.eval()

        self.N = N
        self.cur_count = 0
        # post progress
        self.pred_conf = np.zeros([N, ])
        self.gcn_feature = np.zeros([N, 1024])

    def inference(self, features, knns_origin, idx_origin, n):
        """

        :param features: [M, 512] 全部特征（其中包含N个关注的特征和M-N个邻居涉及特征）
        :param knns_origin: [M, 2, k] 全部knn（带原始下标）
        :param idx_origin: [M,] 每个特征在原始的下标
        :param n: 关注的特征数量(上述输入的*前*n个就是关注的特征，即需要计算的特征)
        :return:
            1. pred_conf: 这n个特征的置信度
            2. gcn_feat: 这n个特征的gcn特征(1024)
        """
        if self.cur_count >= self.N:
            print("all batches are over, use .. to get the final cluster result.")
            return
        assert len(features) == len(knns_origin) == len(idx_origin)
        m = len(features)

        # build index mapping
        # index mapping origin2new & new2origin
        idx_ori2new = dict()
        idx_new2ori = dict()
        for i in range(m):
            idx_ori2new[idx_origin[i]] = i
            idx_new2ori[i] = idx_origin[i]

        # alter knns_origin to knns_new(change the index from old to new)
        knns_new = []
        for knn_o in knns_origin:
            nbrs_o = list(knn_o[0])
            nbrs_new = []
            for nbr in nbrs_o:
                nbrs_new.append(idx_ori2new[nbr])
            knn_n = (np.array(nbrs_new).astype(np.int32), knn_o[1])
            knns_new.append(knn_n)

        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        Adj = fast_knns2spmat(knns_new, self.k, self.cut_edge_sim_th, use_sim=True)
        # build symmetric adjacency matrix
        Adj = build_symmetric_adj(Adj, self_loop=True)  # 加上自身比较 相似度1
        Adj = row_normalize(Adj)  # 归一化

        output, gcn_feat = self.model(features, Adj, output_feat=True)

        pred_confs = output.detach().cpu().numpy()
        gcn_feat = gcn_feat.detach().cpu().numpy()

        self.gcn_feature[idx_origin[:n]] = gcn_feat[:n]
        self.pred_conf[idx_origin[:n]] = pred_confs[:n]

        self.cur_count += n

    def get_cluster_result(self):
        if self.cur_count < self.N:
            print("the batch inference is not over")
            return

        self.gcn_feature = l2norm(self.gcn_feature)
        knns = build_knns(self.gcn_feature, self.knn_method, self.k)

        dists, nbrs = knns2ordered_nbrs(knns)
        pred_dist2peak, pred_peaks = confidence_to_peaks(dists, nbrs, self.pred_conf, self.max_conn)
        pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, self.tau_gcn, self.N)
        return pred_labels
