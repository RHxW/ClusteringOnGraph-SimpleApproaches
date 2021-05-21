import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, feature_dim, embed_dim, gcn=False):
        super(Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feature_dim if gcn else 2 * feature_dim))
        nn.init.xavier_uniform(self.weight)

    def forward(self, features, nodes, adj):
        pass