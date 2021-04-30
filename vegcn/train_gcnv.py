import torch
import os
from vegcn.models.gcn_v import GCN_V
from vegcn.config.gcnv_config import CONFIG
from vegcn.dataset.gcn_v_dataset import GCNVDataset
from utils import sparse_mx_to_torch_sparse_tensor


def train_gcnv(cfg):
    device = cfg["device"]
    # dataset
    cfg["phase"] = "train"
    dataset = GCNVDataset(cfg)

    # model
    feature_dim = cfg["feature_dim"]
    nhid = cfg["nhid"]
    # nlayer = cfg["nlayer"]
    nclass = cfg["nclass"]
    dropout = cfg["dropout"]
    model = GCN_V(feature_dim, nhid, nclass, dropout).to(device)
    # load checkpoint
    checkpoint_path = cfg["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    # optimizer
    lr = cfg["lr"]
    momentum = cfg["momentum"]
    weight_decay = cfg["weight_decay"]
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # loss func
    loss_func = torch.nn.MSELoss()

    # train data
    features = torch.tensor(dataset.features, dtype=torch.float32).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(dataset.adj).to(device)
    labels = torch.tensor(dataset.labels, dtype=torch.float32).to(device)

    output_feature = False

    epochs = cfg["epochs"]
    for epoch in range(epochs):
        pred, out_feat = model(features, adj, output_feature)
        loss = loss_func(pred, labels)
        loss_val = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: %d, loss %.4f" % (epoch, loss_val))

    torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    cfg = CONFIG
    train_gcnv(cfg)