import os
import numpy as np
import torch
from scipy.sparse import csr_matrix

from STAR_FC.config.starfc_config import CONFIG
from STAR_FC.dataset.starfc_dataset import STARFCDataset, SRStrategyClass
from STAR_FC.models.gcn_starfc import GCN_STARFC, ClassifierHead
from utils import (l2norm, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values, sparse_mx_to_torch_sparse_tensor,
                   intdict2ndarray, Timer)
from utils.get_knn import build_knns
from utils.deduce import edge_to_connected_graph

from evaluation.Purity_Diverse_V import get_DPV_measure
from evaluation.metrics import pairwise


def test_starfc(cfg, edge_pred_batch_num):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

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

    knn = build_knns(feature, knn_method, k)
    knn = np.array(knn)
    inst_num = len(knn)
    nbrs = knn[:, 0, :].astype(np.int32)
    dists = knn[:, 1, :].astype(np.float32)

    Adj = fast_knns2spmat(knn, k, cut_edge_sim_th, use_sim=True)
    # build symmetric adjacency matrix
    Adj = build_symmetric_adj(Adj, self_loop=True)
    Adj = row_normalize(Adj)
    Adj = sparse_mx_to_torch_sparse_tensor(Adj)

    # pair_a 与 pair_b共同构成边的信息
    pair_a = []
    pair_b = []
    pair_a_new = []
    pair_b_new = []
    for i in range(inst_num):
        pair_a.extend([int(i)] * (len(nbrs[i]) - 1))
        pair_b.extend([int(j) for j in nbrs[i][1:]])  # 去掉指向自身的边
    pair_a = np.array(pair_a)
    pair_b = np.array(pair_b)
    # for i in range(len(pair_a)):
    #     if pair_a[i] != pair_b[i]:
    #         # 去掉指向自身的那条边（至于这么麻烦么？）
    #         pair_a_new.extend([pair_a[i]])
    #         pair_b_new.extend([pair_b[i]])
    # pair_a = np.array(pair_a_new)
    # pair_b = np.array(pair_b_new)

    feature = torch.from_numpy(feature).to(device)
    Adj = Adj.to(device)
    x = model_gcn(feature, Adj)

    softmax = torch.nn.Softmax(dim=1)
    # 分batch操作（边太多，特征太大存不下）
    edges = []
    ep_batch_size = int(len(pair_a) / edge_pred_batch_num)
    for i in range(edge_pred_batch_num):
        id_a = pair_a[i * ep_batch_size: (i + 1) * ep_batch_size]
        id_b = pair_b[i * ep_batch_size: (i + 1) * ep_batch_size]
        new_feat = torch.cat([x[id_a], x[id_b]], 1)
        pred = model_ch(new_feat)
        pred = softmax(pred)
        score_ = pred[:, 1].cpu().numpy()
        idx = np.where(score_ > threshold1)[0].tolist()
        id1 = np.array([pair_a[idx].tolist()])
        id2 = np.array([pair_b[idx].tolist()])
        edges.extend(np.concatenate([id1, id2], 0).transpose().tolist())

    # 2. Graph refinement: try to identify the edges which can not be removed directly using edge scores with node intimacy (NI)
    # use node intimacy to represent the edge score and remove those edges whose score is below t2
    edges_count = len(edges)
    value = [1] * edges_count
    edges = np.array(edges)
    adj2 = csr_matrix((value, (edges[:, 0].tolist(), edges[:, 1].tolist())), shape=(edges_count, edges_count))
    link_num = np.array(adj2.sum(axis=1))
    common_link = adj2.dot(adj2)  # common_link中的每个元素aij代表第i节点和第j节点的共同节点的个数

    edges_new = []
    share_num = common_link[edges[:, 0].tolist(), edges[:, 1].tolist()].tolist()[0]
    edges = edges.tolist()

    for i in range(len(edges)):
        if link_num[edges[i][0]] * link_num[edges[i][1]] != 0:  # 不是孤立节点（孤立节点怎么处理）
            if max((share_num[i]) / link_num[edges[i][0]], (share_num[i]) / link_num[edges[i][1]]) > threshold2:
                edges_new.append(edges[i])

    pred_labels = edge_to_connected_graph(edges_new, inst_num)

    # evaluation
    lb2idxs, idx2lb = dataset.lb2idxs, dataset.idx2lb
    inst_num = len(idx2lb)  # 样本数量
    label_true = intdict2ndarray(idx2lb)  # 真实标签

    avg_pre, avg_rec, fscore = pairwise(label_true, pred_labels)
    print("result class count: %d." % len(set(pred_labels)))
    print("pairwise F-score: avg_pre: %.6f, avg_rec: %.6f, fscore: %.6f" % (avg_pre, avg_rec, fscore))
    # V-measure
    diverse_score, purity_score, V_measure = get_DPV_measure(label_true, pred_labels)
    h, c, v = V_measure
    print("V-measure score: h: %.6f, c: %.6f, v: %.6f" % (h, c, v))
    print("*" * 50)

if __name__ == "__main__":
    cfg = CONFIG
    edge_pred_batch_num = 10
    test_starfc(cfg, edge_pred_batch_num)
