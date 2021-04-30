import torch.nn as nn
from collections import OrderedDict
from gcn_models.utils import GraphConv, MeanAggregator


class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_V, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        # for i in range(1, nlayer):
        #     self.add_module("conv%d" % i, GraphConv(nhid, nhid, MeanAggregator, dropout))

        self.nclass = nclass
        self.classifier = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.PReLU(nhid),
            nn.Linear(nhid, self.nclass)
        )

    def forward(self, x, adj, output_feat=False):
        x = self.conv1(x, adj)
        pred = self.classifier(x).view(-1)

        if output_feat:
            return pred, x
        return pred, None
