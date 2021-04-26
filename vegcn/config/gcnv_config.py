CONFIG = {
    "proj_name": "train_gcnv_1",
    "phase": "train",
    "train_data_root": "/tmp/pycharm_project_444/data/train/",
    "test_data_root": "",
    "device": "cuda:0",
    "checkpoint_path": "/tmp/pycharm_project_444/data/train/train_gcnv_1/checkpoint_1.pth",

    "epochs": 20,

    "knn": 20,
    "knn_method": "faiss",  # nmslib, faiss-gpu
    "cut_edge_sim_th": 0.6,  # origin th_sim
    "eval_interim": True,  # ?

    "feature_dim": 512,
    "is_norm_feat": True,
    "nhid": 1024,
    "nlayer": 1,
    "nclass": 1,
    "dropout": 0.,
    "use_gcn_feat": True,
    "max_conn": 10,
    "tau_0": 0.67,  # origin feature tau
    "tau": 0.67,  # gcn feature tau

    # optimizer
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-5,

    "conf_metric": "s_nbr",
    "save_decomposed_adj": False, # if True: sparse adjacency matrix ---> indices, values, shape
}