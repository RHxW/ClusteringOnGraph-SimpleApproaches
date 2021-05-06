import os
import numpy as np
import torch

from STAR_FC.config.starfc_config import CONFIG
from STAR_FC.dataset.starfc_dataset import STARFCDataset, SRStrategyClass
from STAR_FC.models.gcn_starfc import GCN_STARFC, ClassifierHead


def test_starfc(cfg):
    pass