import os
import numpy as np
import random

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values,
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
        self.M = cfg["M"]
        self.N = cfg["N"]
        self.K1 = cfg["K1"]
        self.K2 = cfg["K2"]
        self.sample_ratio = cfg["sample_ratio"]

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

        # with Timer('read knn graph'):
        #     if os.path.isfile(self.knn_graph_path):
        #         print("load knns from the knn_path")
        #         self.knns = np.load(self.knn_graph_path)['data']
        #     else:
        #         if self.knn_graph_path is not None:
        #             print('knn_graph_path does not exist: {}'.format(self.knn_graph_path))
        #         knn_prefix = os.path.join(cfg.prefix, 'knns', cfg.name)
        #         self.knns = build_knns(knn_prefix, self.features, cfg.knn_method, cfg.knn)
        #
        #     adj = fast_knns2spmat(self.knns, self.k, self.cut_edge_sim_th, use_sim=True)
        #
        #     # build symmetric adjacency matrix
        #     adj = build_symmetric_adj(adj, self_loop=True)
        #     # print('adj before norm')
        #     # print(adj)
        #     adj = row_normalize(adj)
        #     if self.save_decomposed_adj:
        #         adj = sparse_mx_to_indices_values(adj)
        #         self.adj_indices, self.adj_values, self.adj_shape = adj
        #     else:
        #         self.adj = adj
        #
        #     # convert knns to (dists, nbrs)
        #     self.dists, self.nbrs = knns2ordered_nbrs(self.knns)

        print('feature shape: {}, k: {}, norm_feat: {}'.format(self.features.shape, self.k, self.is_norm_feat))

    def __getitem__(self, idx):
        pass

    def get_SPSS_subgraph(self):
        # within the SPSS(Structure-Preserved Subgraph Sampling) procedure
        # 1 select M seed clusters
        seed_clusters = random.sample(self.class_ids, self.M)
        center_features = self.center_feat[seed_clusters]
        # 2 select M seed_clusters's N-nearest neighbour -> S1(M*N)
        S1 = set()  # class id set

    def __len__(self):
        pass