import os
import numpy as np

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values,
                   intdict2ndarray, Timer)
from vegcn.confidence import (confidence, confidence_to_peaks)


class GCNVDataset(object):
    def __init__(self, cfg):
        self.train_data_root = cfg["train_data_root"]
        self.proj_name = cfg["proj_name"]
        self.proj_path = os.path.join(self.train_data_root, self.proj_name)
        self.feat_path = os.path.join(self.proj_path, "feature.npy")
        self.label_path = os.path.join(self.proj_path, "label.txt")
        self.knn_graph_path = os.path.join(self.proj_path, "%s_k_%d.npz" % (cfg["knn_method"], cfg["knn"]))

        self.k = cfg['knn']
        self.knn_method = cfg["knn_method"]
        self.feature_dim = cfg["feature_dim"]
        self.is_norm_feat = cfg["is_norm_feat"]
        self.save_decomposed_adj = cfg["save_decomposed_adj"]

        self.cut_edge_sim_th = cfg["cut_edge_sim_th"]
        self.max_conn = cfg["max_conn"]
        self.conf_metric = cfg["conf_metric"]

        with Timer('read meta and feature'):
            if os.path.exists(self.label_path):
                self.lb2idxs, self.idx2lb = read_meta(self.label_path)
                self.inst_num = len(self.idx2lb)    # 样本数量
                self.gt_labels = intdict2ndarray(self.idx2lb)   # 真实标签
                self.ignore_label = False
            else:
                self.inst_num = -1
                self.ignore_label = True

            if not os.path.exists(self.feat_path):
                raise RuntimeError("feat_path not exists!!!")
            self.features = read_probs(self.feat_path, self.inst_num,
                                       self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = 1  # take the entire graph as input

        with Timer('read knn graph'):
            if os.path.exists(self.knn_graph_path):
                knns = np.load(self.knn_graph_path)['data']
            else:
                print('knn_graph_path does not exist: {}'.format(self.knn_graph_path))
                # knn_prefix = os.path.join(cfg.prefix, 'knns', cfg.test_name)
                # knn_prefix = cfg.knn_graph_path.split('faiss_k')[0]
                knn_prefix = self.proj_path
                # cfg.knn = 5
                # cfg.knn_method = 'faiss'
                knns = build_knns(knn_prefix, self.features, self.knn_method,
                                  self.k)  # shape=(26960, 2, 5)

            adj = fast_knns2spmat(knns, self.k, self.cut_edge_sim_th, use_sim=True)

            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)  # 加上自身比较 相似度1
            adj = row_normalize(adj)    # 归一化
            if self.save_decomposed_adj:
                adj = sparse_mx_to_indices_values(adj)
                self.adj_indices, self.adj_values, self.adj_shape = adj
            else:
                self.adj = adj

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns)

        print('feature shape: {}, k: {}, norm_feat: {}'.format(
            self.features.shape, self.k, self.is_norm_feat))

        if not self.ignore_label:
            with Timer('Prepare ground-truth label'):
                self.labels = confidence(feats=self.features,
                                         dists=self.dists,
                                         nbrs=self.nbrs,
                                         metric=self.conf_metric,
                                         idx2lb=self.idx2lb,
                                         lb2idxs=self.lb2idxs)
                if cfg.eval_interim:
                    _, self.peaks = confidence_to_peaks(self.dists, self.nbrs, self.labels, self.max_conn)

    def __getitem__(self, index):
        ''' return the entire graph for training.
        To accelerate training or cope with larger graph,
        we can sample sub-graphs in this function.
        '''
        ## TODO! NO USE
        assert index == 0
        return (self.features, self.adj_indices, self.adj_values,
                self.adj_shape, self.labels)

    def __len__(self):
        return self.size
