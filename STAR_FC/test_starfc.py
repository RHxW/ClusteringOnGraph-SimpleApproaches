import os
import numpy as np
import torch

from STAR_FC.config.starfc_config import CONFIG
from STAR_FC.dataset.starfc_dataset import STARFCDataset, SRStrategyClass
from STAR_FC.models.gcn_starfc import GCN_STARFC, ClassifierHead
from utils import (l2norm, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values, sparse_mx_to_torch_sparse_tensor,
                   intdict2ndarray, Timer)
from utils.knn import build_knns_simple


def test_starfc(cfg):
    # 1. Graph parsing: feed the *entire* graph into the GCN and obtain all edge scores simultaneously
    #  perform simple but effective pruning with a single threshold t1
    cfg["phase"] = "test"
    dataset = STARFCDataset(cfg)
    device = cfg["device"]

    # model
    feature_dim = cfg["feature_dim"]
    nhid = cfg["nhid"]
    model_gcn = GCN_STARFC(feature_dim, nhid).to(device)
    model_ch = ClassifierHead(nhid).to(device)

    gcn_checkpoint_path = cfg["gcn_checkpoint_path"]
    if os.path.exists(gcn_checkpoint_path):
        model_gcn.load_state_dict(torch.load(gcn_checkpoint_path))
    else:
        raise RuntimeError("gcn checkpoint not exists!!!")
    ch_checkpoint_path = cfg["ch_checkpoint_path"]
    if os.path.exists(ch_checkpoint_path):
        model_ch.load_state_dict(torch.load(ch_checkpoint_path))
    else:
        raise RuntimeError("classifier checkpoint not exists!!!")

    feature = dataset.features
    gt_label = dataset.gt_labels

    k = cfg['knn']
    knn_method = cfg["knn_method"]
    cut_edge_sim_th = cfg["cut_edge_sim_th"]

    threshold1 = cfg["threshold1"]
    threshold2 = cfg["threshold2"]

    knn = build_knns_simple(feature, knn_method, k)
    inst_num = len(knn)
    nbrs = knn[:, 0, :]
    dists = knn[:, 1, :]

    Adj = fast_knns2spmat(knn, k, cut_edge_sim_th, use_sim=True)
    # build symmetric adjacency matrix
    Adj = build_symmetric_adj(Adj, self_loop=True)
    Adj = row_normalize(Adj)
    Adj = sparse_mx_to_torch_sparse_tensor(Adj)

    pair_a = []
    pair_b = []
    pair_a_new = []
    pair_b_new = []
    for i in range(inst_num):
        pair_a.extend([int(i)] * k)
        pair_b.extend([int(j) for j in nbrs[i]])
    for i in range(len(pair_a)):
        if pair_a[i] != pair_b[i]:
            pair_a_new.extend([pair_a[i]])
            pair_b_new.extend([pair_b[i]])
    pair_a = np.array(pair_a_new)
    pair_b = np.array(pair_b_new)

    x = model_gcn(feature, Adj)
    pred = model_ch(x)
    softmax = torch.nn.Softmax(dim=1)
    pred = softmax(pred)
    score_ = pred[:,1]
    idx = np.where(score_ > threshold1)[0].tolist()

    id1 = np.array([pair_a[idx].tolist()])
    id2 = np.array([pair_b[idx].tolist()])
    edges = np.concatenate([id1, id2], 0).transpose().tolist()

    # 2. Graph refinement: try to identify the edges which can not be removed directly using edge scores with node intimacy (NI)
    # use node intimacy to represent the edge score and remove those edges whose score is below t2
    pass


if __name__ == "__main__":
    cfg = CONFIG
    test_starfc(cfg)