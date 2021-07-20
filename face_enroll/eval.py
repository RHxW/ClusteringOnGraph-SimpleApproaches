import os
import numpy as np
from scipy import sparse as sp

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)  # 用于构建稀疏矩阵，第一个参数为(value, (row, col))
    if sparse:
        contingency = contingency.tocsr()  # 稀疏矩阵形式转换成一般矩阵形式
        contingency.sum_duplicates()  # 把0去掉，按行列顺序输出
        # array([[4, 0, 0, 1],
        #        [0, 3, 0, 1],     -->   array([4, 1, 3, 1, 2, 1, 1], dtype=int32)
        #        [0, 0, 2, 1],
        #        [0, 0, 0, 1]])
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def fscore(labels_true, labels_pred, sparse=False):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples, = labels_true.shape

    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    # array([[4, 0, 0, 1],
    #        [0, 3, 0, 1],     -->   array([4, 1, 3, 1, 2, 1, 1], dtype=int32)
    #        [0, 0, 2, 1],
    #        [0, 0, 0, 1]])
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)
    return 100 * avg_pre, 100 * avg_rec, 100 * fscore


def get_label_true(images_root):
    if not os.path.exists(images_root):
        return None
    if images_root[-1] != "/":
        images_root += "/"

    ids = os.listdir(images_root)
    labels = dict()
    for _id in ids:
        imgs = os.listdir(images_root + _id + "/")
        for _img in imgs:
            _img_name = os.path.splitext(_img)[0]
            labels[_img_name] = int(_id)

    ks = list(labels.keys())
    ks.sort()
    label_true = []
    for k in ks:
        label_true.append(labels[k])

    return label_true, ks

def get_label_pred(pred_root, keys):
    if not os.path.exists(pred_root):
        return None
    if pred_root[-1] != "/":
        pred_root += "/"

    ids = os.listdir(pred_root)
    labels = dict()
    for _id in ids:
        if _id in ["features"]:
            continue
        imgs = os.listdir(pred_root + _id + "/")
        for _img in imgs:
            _img_name = os.path.splitext(_img)[0]
            _img_name = _img_name.split("_")[-1]
            labels[_img_name] = int(_id)

    label_pred = []
    for k in keys:
        label_pred.append(labels[k])

    return label_pred

def get_label_eval(images_root, pred_root):
    label_true, ks = get_label_true(images_root)
    label_pred = get_label_pred(pred_root, ks)
    label_true = np.array(label_true)
    label_pred = np.array(label_pred)

    valid = np.where(label_pred > -1)
    q_all = np.array(label_pred > -2)

    return label_true, label_pred, valid, q_all