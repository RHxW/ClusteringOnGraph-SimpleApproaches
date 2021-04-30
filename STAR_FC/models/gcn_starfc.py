from torch import nn

from gcn_models.utils import GraphConv, MeanAggregator


class GCN_STARFC(nn.Module):
    def __init__(self, feature_dim, nhid, nlayer, nclass, dropout=0):
        super(GCN_STARFC, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        for i in range(1, nlayer):
            self.add_module("conv%d" % (i + 1), GraphConv(nhid, nhid, MeanAggregator, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(nhid * 2, nhid),
            nn.PReLU(nhid),
            nn.Linear(nhid, 2)
        )

    def forward(self, x, adj, output_feat=False):
        x = self.conv1(x, adj)
        pred = self.classifier(x).view(-1)

        if output_feat:
            return pred, x
        return pred, None
