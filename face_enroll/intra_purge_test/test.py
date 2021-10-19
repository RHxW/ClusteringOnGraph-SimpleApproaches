import os
import shutil
import numpy as np

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec

def test():
    root_path = "F:/COGSAs/ClusteringOnGraph-SimpleApproaches/face_enroll/intra_purge_test/"
    data_root = root_path + "test_data/"
    pic_root = data_root + "pics/"
    feat_root = data_root + "feats/"
    res_root = root_path + "test_res/"
    if os.path.exists(res_root):
        shutil.rmtree(res_root)
    os.mkdir(res_root)

    img_files = os.listdir(pic_root)
    feat_files = os.listdir(feat_root)
    if len(img_files) != len(feat_files):
        print(len(img_files), len(feat_files))
        return

    fl = []
    for ff in feat_files:
        fl.append(np.fromfile(feat_root + ff, dtype=np.float32))
    features_all = np.array(fl)
    fa = l2norm(features_all)
    dists = np.matmul(fa, fa.T)

    return dists



