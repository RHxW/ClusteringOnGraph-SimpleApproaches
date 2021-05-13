import numpy as np

__all__ = ['peaks_to_labels']


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):  # 没看懂~~~~~
        idx.append(u)
        u = parent[u]

        # parent[u] = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


def peaks_to_edges(peaks, dist2peak, tau):
    edges = []
    for src in peaks:
        dsts = peaks[src]
        dists = dist2peak[src]
        for dst, dist in zip(dsts, dists):
            if src == dst or dist >= 1 - tau:
                continue
            edges.append([src, dst])
    return edges


def peaks_to_labels(peaks, dist2peak, tau, inst_num):
    # peaks是每个节点邻居中confidence（中心概率）比它大的节点的idx，dist2peak是对应的距离
    # peaks其实就是edges，但还要根据一个阈值tau进行排除，排除后得到最终的edges
    edges = peaks_to_edges(peaks, dist2peak, tau)
    pred_labels = edge_to_connected_graph(edges, inst_num)
    return pred_labels
