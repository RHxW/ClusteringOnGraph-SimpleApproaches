import numpy as np

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])  # embedding中特征点对数是否相等
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])  # lfw中点对的个数，共有6000对
    nrof_thresholds = len(thresholds)  # thresholds是一个长度为400的数组，从0开始，间隔0.01
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))  # ptrs是一个10x400的数组
    fprs = np.zeros((nrof_folds, nrof_thresholds))  # fprs是一个10x400的数组
    accuracy = np.zeros((nrof_folds))  # accuracy是一个1x10的数组
    indices = np.arange(nrof_pairs)  # indices的范围为(0, 6000)
    # print('pca', pca)

    if pca == 0:  # 是否使用pca进行压缩
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)  # 计算特征点之间的距离

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):  # 将数据分为10份，1份是测试，9份是训练
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:  # 如果使用了pca进行压缩
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)  # 从pca中解压出真正的特征值embed
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)  # 计算特征点之间的距离
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):  # 在0-4中，间隔0.01，找到其中acc_train最大的值作为阈值
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)  # 得到最好的 threshold_index
        # print('threshold', thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        # 根据上面获取的最佳阈值，求测试数据集的准确率，作为最终的准确率
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)  # 由于还有交叉验证，因此还需要求10次tprs和fprs数组中的均值
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))  # 真正
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))  # 假正
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))  # 真负
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))  # 假负

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc