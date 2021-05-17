import os
import torch
import numpy as np

from vegcn.models.gcn_v import GCN_V
from vegcn.confidence import confidence_to_peaks
from vegcn.deduce import peaks_to_labels

from utils import (sparse_mx_to_torch_sparse_tensor, list2dict,
                   intdict2ndarray, knns2ordered_nbrs, Timer, read_probs, l2norm, fast_knns2spmat,
                   build_symmetric_adj, row_normalize)
from utils.get_knn import build_knns


def inference_gcnv(cfg, feature_path):
    pass