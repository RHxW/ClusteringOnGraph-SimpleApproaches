import torch
import torch.nn as nn
import random

from utils import neighbour_sample

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, nodes, nbrs, num_sample=10, add_self_loop=False):
        mask, unique_nodes_list = neighbour_sample(nodes, nbrs, num_sample, add_self_loop)
        embedding_matrix = features[unique_nodes_list]
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)  # mask归一化
        agg_features = mask.mm(embedding_matrix)
        return agg_features

class MaxPoolAggregator(nn.Module):
    def __init__(self):
        super(MaxPoolAggregator, self).__init__()

    def forward(self, features, nodes, nbrs, num_sample=10):
        mask, unique_nodes_list = neighbour_sample(nodes, nbrs, num_sample)
        embedding_matrix = features[unique_nodes_list]
        idxs = [x.nonzero() for x in mask == 1]  # mask中为1的位置的列下标，按行输出
        agg_features = []
        for idx in idxs:
            embds = embedding_matrix[idx.squeeze()]
            for embd in embds:
                if embd.dim() == 1:
                    agg_features.append(embd.view(1, -1))
                else:
                    agg_features.append(torch.max(embd, 0).values.view(1,-1))
        agg_features = torch.cat(agg_features, 0)
        return agg_features
