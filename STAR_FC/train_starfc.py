import os
import numpy as np
import torch

from STAR_FC.config.starfc_config import CONFIG
from STAR_FC.dataset.starfc_dataset import STARFCDataset, SRStrategyClass


def test_starfc(cfg):
    dataset = STARFCDataset(cfg)
    S2_features, S2_adj, S2_lb2idx, S2_idx2lb, S2_label = dataset.get_SPSS_subgraph()
    SRSC = SRStrategyClass(S2_features, S2_adj, S2_lb2idx, S2_idx2lb, S2_label, dataset.K2_ratio)
    print(dataset)

    loss_func = torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    cfg = CONFIG
    test_starfc(cfg)