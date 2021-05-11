from torch import nn

from gcn_models.utils import GraphConv, MeanAggregator


class GCN_STARFC(nn.Module):
    def __init__(self, feature_dim, nhid, dropout=0):
        super(GCN_STARFC, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        # for i in range(1, nlayer):
        #     self.add_module("conv%d" % (i + 1), GraphConv(nhid, nhid, MeanAggregator, dropout))

        # self.classifier = nn.Sequential(
        #     nn.Linear(nhid * 2, nhid),
        #     nn.PReLU(nhid),
        #     nn.Linear(nhid, 2)
        # )

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, nhid):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(nhid * 2, nhid),
            nn.PReLU(nhid),
            nn.Linear(nhid, 2)
        )

    def forward(self, x):
        pred = self.classifier(x)
        return pred