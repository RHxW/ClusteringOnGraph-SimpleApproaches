import os
import numpy as np
import torch

from STAR_FC.config.starfc_config import CONFIG
from STAR_FC.dataset.starfc_dataset import STARFCDataset, SRStrategyClass
from STAR_FC.models.gcn_starfc import GCN_STARFC, ClassifierHead


def train_starfc(cfg):
    dataset = STARFCDataset(cfg)
    device = cfg["device"]

    # model
    feature_dim = cfg["feature_dim"]
    nhid = cfg["nhid"]
    model_gcn = GCN_STARFC(feature_dim, nhid).to(device)
    model_ch = ClassifierHead(nhid).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    gcn_checkpoint_path = cfg["gcn_checkpoint_path"]
    if os.path.exists(gcn_checkpoint_path):
        model_gcn.load_state_dict(torch.load(gcn_checkpoint_path))
    ch_checkpoint_path = cfg["ch_checkpoint_path"]
    if os.path.exists(ch_checkpoint_path):
        model_ch.load_state_dict(torch.load(ch_checkpoint_path))

    lr = cfg["lr"]
    momentum = cfg["momentum"]
    weight_decay = cfg["weight_decay"]
    optimizer = torch.optim.SGD([{'params': model_gcn.parameters(), 'weight_decay': weight_decay},
                                 {'params': model_ch.parameters(), 'weight_decay': weight_decay}], lr=lr,
                                momentum=momentum)

    epochs = cfg["epochs"]
    SR_epochs = cfg["SR_epochs"]
    for epoch in range(epochs):
        S2_features, S2_adj, S2_lb2idx, S2_idx2lb, S2_label = dataset.get_SPSS_subgraph()
        S2_features = torch.from_numpy(S2_features).to(device)
        S2_adj = S2_adj.to(device)
        SRSC = SRStrategyClass(S2_adj, S2_lb2idx, S2_idx2lb, S2_label, dataset.K2_ratio)
        # Sample Randomness Procedure
        for sre in range(SR_epochs):
            x = model_gcn(S2_features, S2_adj)
            S_feature, S_label = SRSC.get_Subgraph(x)
            S_pred = model_ch(S_feature)  # obtain the edge scores
            S_label = torch.from_numpy(S_label).to(torch.long).to(device)
            loss = loss_func(S_pred, S_label)
            loss_val = loss.item()
            print("EPOCH: %d \t| sr epoch: %d \t| loss: %.6f" % (epoch, sre, loss_val))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model_gcn.state_dict(), gcn_checkpoint_path)
    torch.save(model_ch.state_dict(), ch_checkpoint_path)


if __name__ == "__main__":
    cfg = CONFIG
    train_starfc(cfg)
