import random
import torch


class Sampler():
    def __init__(self) -> None:
        pass

def neighbour_sample(nodes, nbrs, num_sample=10, add_self_loop=False):
    assert len(nodes) == len(nbrs)
    if num_sample < 1:
        raise RuntimeError("num_sample error!!!")
    elif num_sample is not None:
        sample_nbrs = []
        for sub_nbrs in nbrs:
            if len(sub_nbrs) > num_sample:
                # 随机采样
                sample_nbrs.append(set(random.sample(sub_nbrs, num_sample)))
            else:
                sample_nbrs.append(sub_nbrs)
    else:
        sample_nbrs = nbrs
    n = len(nodes)  # 待采样节点数量
    if add_self_loop:
        for i in range(n):
            sample_nbrs[i].add(nodes[i])

    unique_nodes_list = list(set.union(*sample_nbrs))
    m = len(unique_nodes_list)  # 涉及到的全部结点（待采样+相关邻居）数量
    unique_nodes_idx = {n:i for i, n in enumerate(unique_nodes_list)}  # 为了获取col的idx
    mask = torch.zeros([n, m])
    
    col_idxs = []
    row_idxs = []
    for i in range(n):
        sample_nbr = sample_nbrs[i]
        for node in sample_nbr:
            col_idxs.append(unique_nodes_idx[node])
            row_idxs.append(i)

    mask[row_idxs, col_idxs] = 1
    return mask, unique_nodes_list

    
    

