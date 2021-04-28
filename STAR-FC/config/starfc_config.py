CONFIG = {
    "proj_name": "train_gcn_nlayer_1_360k_15000",
    "phase": "train",
    "data_root": "/tmp/pycharm_project_444/data/",
    "device": "cuda:0",
    "checkpoint_path": "/tmp/pycharm_project_444/data/train_gcn_nlayer_1_360k_15000/checkpoint_1.pth",

    "save_output": True,
    "output_root": "/tmp/pycharm_project_444/data/output/",

    "epochs": 20,

    "knn": 20,
    "knn_method": "hnsw",  # hnsw=nmslib, faiss, faiss_gpu
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