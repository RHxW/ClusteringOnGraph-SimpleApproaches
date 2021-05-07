import os
import numpy as np
import torch

from STAR_FC.config.starfc_config import CONFIG
from STAR_FC.dataset.starfc_dataset import STARFCDataset, SRStrategyClass
from STAR_FC.models.gcn_starfc import GCN_STARFC, ClassifierHead


def test_starfc(cfg):
    # 1. Graph parsing: feed the entire graph into the GCN and obtain all edge scores simultaneously
    #  perform simple but effective pruning with a single threshold t1

    # 2. Graph refinement: try to identify the edges which can not be removed directly using edge scores with node intimacy (NI)
    # use node intimacy to represent the edge score and remove those edges whose score is below t2
    pass