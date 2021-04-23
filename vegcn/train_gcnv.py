import torch
import numpy as np

from vegcn.models.gcn_v import GCN_V
from vegcn.config.gcnv_config import CONFIG
from vegcn.dataset.gcn_v_dataset import GCNVDataset
from utils import sparse_mx_to_torch_sparse_tensor


def train_gcnv(cfg):
    device = cfg["device"]
    # dataset
    dataset = GCNVDataset(cfg)

    # model
    feature_dim = cfg["feature_dim"]
    nhid = cfg["nhid"]
    nclass = cfg["nclass"]
    dropout = cfg["dropout"]
    model = GCN_V(feature_dim, nhid, nclass, dropout).to(device)

    # optimizer
    lr = cfg["lr"]
    momentum = cfg["momentum"]
    weight_decay = cfg["weight_decay"]
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # train data
    features = torch.tensor(dataset.features, dtype=torch.float32).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(dataset.adj).to(device)
    labels = torch.tensor(dataset.labels, dtype=torch.float32).to(device)
    train_data = [features, adj, labels]
