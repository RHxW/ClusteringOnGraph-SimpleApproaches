import numpy as np
import os
import shutil

def feature_to_bin(bin_path, feats):
    # 特征保存成.bin文件供聚类算法读取
    np.array(feats, dtype=np.float32).tofile(bin_path)

def get_list(list_path, paths):
    with open(list_path, "w", encoding="utf-8") as f:
        for img in paths:
            f.write(img + "\n")

def get_pics_id_dir(id_root):
    # 读取测试数据
    if id_root and id_root[-1] != "/":
        id_root += "/"
    ids = os.listdir(id_root)
    imgs = []
    for id_dir in ids:
        id_path = id_root + id_dir + "/"
        if not os.path.isdir(id_path):
            continue
        id_imgs = os.listdir(id_path)
        for _img in id_imgs:
            img_path = id_path + _img
            imgs.append(img_path)

    return imgs

def dir_copy(dir_path, dst_path):
    # 将dir_path下的内容整体复制到dst_path下
    if not os.path.exists(dir_path):
        return
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    if dir_path[-1] != "/":
        dir_path += "/"
    if dst_path[-1] != "/":
        dst_path += "/"
    files = os.listdir(dir_path)
    for file in files:
        if os.path.isdir(dir_path + file):
            dir_copy(dir_path + file + "/", dst_path + file + "/")
        else:
            shutil.copy(dir_path + file, dst_path + file)

def get_singularity_dirs(DB_root, singularity_dir_name="-1"):
    if not os.path.exists(DB_root):
        return
    if DB_root[-1] != "/":
        DB_root += "/"
    singularity_dir = DB_root + singularity_dir_name + "/"
    if not os.path.exists(singularity_dir):
        os.mkdir(singularity_dir)

    dirs = os.listdir(DB_root)
    for _dir in dirs:
        if _dir == singularity_dir_name:
            continue
        _dir_path = DB_root + _dir + "/"
        if len(os.listdir(_dir_path)) <= 1:
            dir_copy(_dir_path, singularity_dir)
            shutil.rmtree(_dir_path)


def get_avg_feature(id_feature_dir):
    if id_feature_dir[-1] != "/":
        id_feature_dir += "/"
    ffiles = os.listdir(id_feature_dir)
    feat = np.zeros(512)
    N = len(ffiles)
    for ff in ffiles:
        feat += np.fromfile(id_feature_dir + ff, dtype=np.float32)

    feat /= N
    return feat

def get_avg_feature_by_list(root, paths):
    if not os.path.exists(root):
        raise RuntimeError("feature root does not exist!!!")
    if root[-1] != "/":
        root += "/"
    feat = np.zeros(512)
    N = len(paths)
    for _path in paths:
        feat += np.fromfile(root + _path, dtype=np.float32)

    feat /= N
    return feat

def get_weight_feature_by_list(root, paths, weight):
    if not os.path.exists(root):
        raise RuntimeError("feature root does not exist!!!")
    if type(weight) not in [list , tuple]:
        raise RuntimeError("weight type error")
    N = len(paths)
    if N < len(weight):  # 如果id内特征不足，则不加权，算简单平均
        return get_avg_feature_by_list(root, paths)
    if np.sum(weight) != 1:  # weight归一化
        weight = list(np.array(weight) / np.sum(weight))
    feat = np.zeros(512)
    for i in range(N):
        _path = paths[i]
        _f = np.fromfile(root + _path, dtype=np.float32)
        _w = weight[i]
        feat += _f * _w
    return feat


def get_list_txt(images_dir, list_txt_path):
    if not os.path.exists(images_dir):
        raise RuntimeError("images_dir not exists!!!")
    if images_dir[-1] != "/":
        images_dir += "/"
    images = os.listdir(images_dir)

    with open(list_txt_path, "w", encoding="utf-8") as f:
        for img in images:
            f.write(images_dir + img + "\n")
