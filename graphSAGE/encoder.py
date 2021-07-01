import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, feature_dim, embed_dim, gcn=False):
        super(Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feature_dim if gcn else 2 * feature_dim))
        nn.init.xavier_uniform(self.weight)

    def forward(self, agg_features, ori_features):
        if not self.gcn:
            combined = torch.cat([ori_features, agg_features], dim=1)
        else:
            combined = agg_features
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
        