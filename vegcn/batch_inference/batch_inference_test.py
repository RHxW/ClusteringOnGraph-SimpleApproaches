import numpy as np

from vegcn.batch_inference.inf_gcnv_batch import GCNVBatchFeeder, GCNVInferenceBatch
from vegcn.config.gcnv_config import CONFIG

def batch_inference(cfg, batch_num, feature_path):
    batch_feeder = GCNVBatchFeeder(cfg, batch_num, feature_path)
    N = batch_feeder.inst_num
    batch_infer = GCNVInferenceBatch(cfg, N)



if __name__ == "__main__":
    feature_path = "/tmp/pycharm_project_444/data/"
    cfg = CONFIG
    batch_num = 10
    batch_inference(cfg, batch_num, feature_path)