import os
import torch

from vegcn.models.gcn_v import GCN_V
from vegcn.confidence import confidence_to_peaks
from vegcn.deduce import peaks_to_labels

from utils import (sparse_mx_to_torch_sparse_tensor, knns2ordered_nbrs, Timer, l2norm, fast_knns2spmat,
                   build_symmetric_adj, row_normalize)
from utils.get_knn import build_knns

class GCNV_API():
    def __init__(self, cfg):
        self.k = cfg['knn']
        self.knn_method = cfg["knn_method"]
        self.is_norm_feat = cfg["is_norm_feat"]

        self.device = cfg["device"]

        self.cut_edge_sim_th = cfg["cut_edge_sim_th"]
        self.max_conn = cfg["max_conn"]
        self.tau_gcn = cfg["tau"]

        # model
        self.feature_dim = cfg["feature_dim"]
        self.nhid = cfg["nhid"]
        self.nclass = cfg["nclass"]
        self.dropout = cfg["dropout"]
        self.model = GCN_V(self.feature_dim, self.nhid, self.nclass, self.dropout).to(self.device)
        # load checkpoint
        checkpoint_path = cfg["checkpoint_path"]
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def cluster(self, features):
        """

        :param features: 输入特征矩阵, np.array([n, 512])
        :return: 输出聚类结果
        """
        if self.is_norm_feat:
            features = l2norm(features)
        inst_num = features.shape[0]

        with Timer('build knn graph'):
            knns = build_knns(features, self.knn_method, self.k)  # shape=(n, 2, k) NEW

        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        # features = torch.from_numpy(features).to(device)
        Adj = fast_knns2spmat(knns, self.k, self.cut_edge_sim_th, use_sim=True)
        # build symmetric adjacency matrix
        Adj = build_symmetric_adj(Adj, self_loop=True)  # 加上自身比较 相似度1
        Adj = row_normalize(Adj)  # 归一化
        Adj = sparse_mx_to_torch_sparse_tensor(Adj).to(self.device)

        output, gcn_features = self.model(features, Adj, output_feat=True)

        pred_confs = output.detach().cpu().numpy()
        gcn_features = gcn_features.detach().cpu().numpy()

        gcn_features = l2norm(gcn_features)
        knns = build_knns(gcn_features, self.knn_method, self.k)

        dists, nbrs = knns2ordered_nbrs(knns)
        pred_dist2peak, pred_peaks = confidence_to_peaks(dists, nbrs, pred_confs, self.max_conn)
        pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, self.tau_gcn, inst_num)
        return pred_labels