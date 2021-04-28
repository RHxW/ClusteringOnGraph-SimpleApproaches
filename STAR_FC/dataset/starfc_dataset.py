import os
import numpy as np
import random

from utils import (read_meta, read_probs, l2norm, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values, sparse_mx_to_torch_sparse_tensor,
                   intdict2ndarray, Timer)
from utils.knn import build_knns_simple

class STARFCDataset():
    def __init__(self, cfg):
        self.phase = cfg["phase"]
        self.data_root = cfg["data_root"]

        self.proj_name = cfg["proj_name"]
        self.proj_path = os.path.join(self.data_root, self.proj_name)
        self.proj_path = os.path.join(self.proj_path, self.phase)
        print("Phase: %s" % self.phase)
        print("Project Path: %s" % self.proj_path)
        print("-" * 50)
        self.feat_path = os.path.join(self.proj_path, "feature.npy")
        self.label_path = os.path.join(self.proj_path, "label.txt")
        # self.knn_graph_path = os.path.join(self.proj_path, "%s_k_%d.npz" % (cfg["knn_method"], cfg["knn"]))

        self.k = cfg['knn']
        self.knn_method = cfg["knn_method"]
        self.feature_dim = cfg["feature_dim"]
        self.is_norm_feat = cfg["is_norm_feat"]
        # self.save_decomposed_adj = cfg["save_decomposed_adj"]

        # spss parameters
        self.M = max(1, cfg["M"])
        self.N = max(1, cfg["N"])
        self.K1_ratio = cfg["K1_ratio"]
        assert 0 < self.K1_ratio <= 1, "K1_ratio error!"
        self.K2_ratio = cfg["K2_ratio"]
        assert 0 < self.K2_ratio <= 1, "K2_ratio error!"

        self.cut_edge_sim_th = cfg["cut_edge_sim_th"]
        # self.max_conn = cfg["max_conn"]

        with Timer('read meta and feature'):
            if os.path.exists(self.label_path):
                self.lb2idxs, self.idx2lb = read_meta(self.label_path)
                self.inst_num = len(self.idx2lb)  # 样本数量
                self.cls_num = len(self.lb2idxs)  # id数量
                self.class_ids = list(self.lb2idxs.keys())  # 所有类id
                assert self.M < self.cls_num, "SPSS parameter M should be smaller than class number!"
                self.gt_labels = intdict2ndarray(self.idx2lb)  # 真实标签
                self.ignore_label = False
            else:
                if self.phase == "train":
                    raise RuntimeError("Training procedure must have label!")
                self.inst_num = -1
                self.cls_num = -1
                self.gt_labels = None
                self.ignore_label = True

            if not os.path.exists(self.feat_path):
                raise RuntimeError("feat_path not exists!!!")
            self.features = read_probs(self.feat_path, self.inst_num, self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            # self.size = 1  # take the entire graph as input

        with Timer('Compute center feature'):
            self.center_feat = np.zeros((self.cls_num, self.features.shape[1])).astype(np.float32)
            lbs = list(self.lb2idxs.keys())  # in case of uncontinuous ids
            for i in range(self.cls_num):
                _id = lbs[i]
                self.center_feat[i] = np.mean(self.features[self.lb2idxs[_id]], 0)
            self.center_feat = l2norm(self.center_feat)

        # construct center_feat knn(k=self.N)
        with Timer("Construct center feature k-NN"):
            self.center_knn = build_knns_simple(self.center_feat, self.knn_method, self.N)

        print('feature shape: {}, k: {}, norm_feat: {}'.format(self.features.shape, self.k, self.is_norm_feat))

    def __getitem__(self, idx):
        pass

    def get_SPSS_subgraph(self):
        # within the SPSS(Structure-Preserved Subgraph Sampling) procedure

        # 1. select M seed clusters from all class
        seed_clusters_idx = random.sample(range(0, self.cls_num), self.M)
        # center_features = self.center_feat[seed_clusters_idx]

        # 2. select M seed_clusters's N-nearest neighbour -> S1(M*N)
        S1 = set()  # class id set
        S1.update(set(seed_clusters_idx))
        for idx in seed_clusters_idx:
            cls_knn = self.center_knn[idx]
            S1.update(set(cls_knn[0]))
        s1n = len(S1)

        # 3. select K1 clusters(idx) from S1 as S2
        K1 = max(int(s1n * self.K1_ratio), 1)
        S2 = set(random.sample(S1, K1))

        # 4. select K2 nodes from S2 as S
        gt_label = self.gt_labels[list(S2)]
        S_lb2idx = dict()
        S_idx2lb = dict()
        S_feat_idxs = set()

        for lb in gt_label:
            _k2 = int(len(self.lb2idxs[lb]) * self.K2_ratio)
            S_lb2idx[lb] = random.sample(self.lb2idxs[lb], _k2)
            for _idx in S_lb2idx[lb]:
                S_idx2lb[_idx] = lb
            S_feat_idxs.update(set(S_lb2idx[lb]))

        S_features = self.features[list(S_feat_idxs)]
        S_node_count = len(S_features)

        # 5. build knn of S
        S_knn = build_knns_simple(S_features, self.knn_method, self.k)
        adj = fast_knns2spmat(S_knn, self.k, self.cut_edge_sim_th, use_sim=True)
        # build symmetric adjacency matrix
        adj = build_symmetric_adj(adj, self_loop=True)
        adj = row_normalize(adj)
        # if self.save_decomposed_adj:
        #     adj = sparse_mx_to_indices_values(adj)
        #     self.adj_indices, self.adj_values, self.adj_shape = adj
        # else:
        #     self.adj = adj
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        return S_features, adj

    def __len__(self):
        pass