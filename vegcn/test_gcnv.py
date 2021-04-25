from __future__ import division

import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F

from vegcn.dataset.gcn_v_dataset import GCNVDataset
from vegcn.confidence import confidence_to_peaks
from vegcn.deduce import peaks_to_labels

from utils import (sparse_mx_to_torch_sparse_tensor, list2dict, write_meta,
                   write_feat, mkdir_if_no_exists, rm_suffix, build_knns,
                   knns2ordered_nbrs, BasicDataset, Timer)
from evaluation import evaluate, accuracy


def test(model, dataset, cfg):
    torch.set_grad_enabled(False)
    features = torch.from_numpy(dataset.features)
    adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)

    device = cfg["device"]
    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    if not dataset.ignore_label:
        labels = torch.from_numpy(dataset.labels)
        labels = labels.to(device)

    model.eval()
    output, gcn_feat = model(features, adj, output_feat=True)
    if not dataset.ignore_label:
        loss = F.mse_loss(output, labels)
        loss_test = float(loss)
        print('[Test] loss = {:.4f}'.format(loss_test))

    pred_confs = output.detach().cpu().numpy()
    gcn_feat = gcn_feat.detach().cpu().numpy()
    return pred_confs, gcn_feat


def test_gcn_v_OLD(model, cfg):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)    # 在对象"cfg.test_data" 末尾追加 "k" 属性及其值v
    dataset = GCNVDataset(cfg)

    folder = '{}_gcnv_k_{}_th_{}'.format(cfg.test_name, cfg.knn, cfg.th_sim)
    oprefix = osp.join(cfg.work_dir, folder)
    # oname = osp.basename(rm_suffix(cfg.load_from))
    oname = 'faiss_k_' + str(cfg.knn)
    opath_pred_confs = osp.join(oprefix, 'pred_confs', '{}.npz'.format(oname))

    if osp.isfile(opath_pred_confs) and not cfg.force:
        data = np.load(opath_pred_confs)
        pred_confs = data['pred_confs']
        inst_num = data['inst_num']
        if inst_num != dataset.inst_num:
            print(
                'WARNING!!! instance number in {} is different from dataset: {} vs {}'.
                format(opath_pred_confs, inst_num, len(dataset)))
    else:
        pred_confs, gcn_feat = test(model, dataset, cfg)
        inst_num = dataset.inst_num

    print('pred_confs: mean({:.4f}). max({:.4f}), min({:.4f})'.format(
        pred_confs.mean(), pred_confs.max(), pred_confs.min()))

    print('Convert to cluster')
    with Timer('Predition to peaks'):
        pred_dist2peak, pred_peaks = confidence_to_peaks(
            dataset.dists, dataset.nbrs, pred_confs, cfg.max_conn)

    if not dataset.ignore_label and cfg.eval_interim:
        # evaluate the intermediate results
        for i in range(cfg.max_conn):
            num = len(dataset.peaks)    # 只用到 num
            pred_peaks_i = np.arange(num)
            peaks_i = np.arange(num)
            for j in range(num):
                if len(pred_peaks[j]) > i:
                    pred_peaks_i[j] = pred_peaks[j][i]
                if len(dataset.peaks[j]) > i:
                    peaks_i[j] = dataset.peaks[j][i]
            acc = accuracy(pred_peaks_i, peaks_i)
            print('[{}-th conn] accuracy of peak match: {:.4f}'.format(
                i + 1, acc))
            acc = 0.
            for idx, peak in enumerate(pred_peaks_i):
                acc += int(dataset.idx2lb[peak] == dataset.idx2lb[idx])
            acc /= len(pred_peaks_i)
            print(
                '[{}-th conn] accuracy of peak label match: {:.4f}'.format(
                    i + 1, acc))

    with Timer('Peaks to clusters (th_cut={})'.format(cfg.tau_0)):
        pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, cfg.tau_0,
                                      inst_num)

    if cfg.save_output:
        print('save predicted confs to {}'.format(opath_pred_confs))
        mkdir_if_no_exists(opath_pred_confs)
        np.savez_compressed(opath_pred_confs,
                            pred_confs=pred_confs,
                            inst_num=inst_num)

        # save clustering results
        idx2lb = list2dict(pred_labels, ignore_value=-1)

        opath_pred_labels = osp.join(
            cfg.work_dir, folder, 'tau_{}_pred_labels.txt'.format(cfg.tau_0))
        print('save predicted labels to {}'.format(opath_pred_labels))
        mkdir_if_no_exists(opath_pred_labels)
        write_meta(opath_pred_labels, idx2lb, inst_num=inst_num)

    # evaluation
    if not dataset.ignore_label:
        print('==> evaluation')
        for metric in cfg.metrics:
            evaluate(dataset.gt_labels, pred_labels, metric)

    if cfg.use_gcn_feat:
        # gcn_feat is saved to disk for GCN-E
        opath_feat = osp.join(oprefix, 'features', '{}.bin'.format(oname))
        if not osp.isfile(opath_feat) or cfg.force:
            mkdir_if_no_exists(opath_feat)
            write_feat(opath_feat, gcn_feat)

        name = rm_suffix(osp.basename(opath_feat))
        prefix = oprefix
        ds = BasicDataset(name=name,
                          prefix=prefix,
                          dim=cfg['nhid'],
                          normalize=True)
        ds.info()

        # use top embedding of GCN to rebuild the kNN graph
        with Timer('connect to higher confidence with use_gcn_feat'):
            knn_prefix = osp.join(prefix, 'knns', name)
            knns = build_knns(knn_prefix,
                              ds.features,
                              cfg["knn_method"],
                              cfg["knn"],
                              is_rebuild=True)
            dists, nbrs = knns2ordered_nbrs(knns)

            pred_dist2peak, pred_peaks = confidence_to_peaks(
                dists, nbrs, pred_confs, cfg.max_conn)
            pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, cfg.tau,
                                          inst_num)

        # save clustering results
        if cfg.save_output:
            oname_meta = '{}_gcn_feat'.format(name)
            opath_pred_labels = osp.join(
                oprefix, oname_meta, 'tau_{}_pred_labels.txt'.format(cfg.tau))
            mkdir_if_no_exists(opath_pred_labels)

            idx2lb = list2dict(pred_labels, ignore_value=-1)
            write_meta(opath_pred_labels, idx2lb, inst_num=inst_num)

        # evaluation
        if not dataset.ignore_label:
            print('==> evaluation')
            for metric in cfg.metrics:
                evaluate(dataset.gt_labels, pred_labels, metric)
