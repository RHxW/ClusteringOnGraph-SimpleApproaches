CONFIG = {
    "proj_name": "train_gcn_nlayer_1_360k_2000",
    "phase": "train",
    "data_root": "/tmp/pycharm_project_444/data/",
    "device": "cuda:0",
    "gcn_checkpoint_path": "/tmp/pycharm_project_444/data/train_gcn_nlayer_1_360k_2000/gcn_checkpoint.pth",
    "ch_checkpoint_path": "/tmp/pycharm_project_444/data/train_gcn_nlayer_1_360k_2000/ch_checkpoint.pth",

    "save_output": True,
    "output_root": "/tmp/pycharm_project_444/data/output/",

    "epochs": 20,
    "SR_epochs": 20,

    "knn": 20,
    "knn_method": "faiss_gpu_single",  # hnsw=nmslib, faiss, faiss_gpu_single, faiss_gpu_all
    "cut_edge_sim_th": 0.6,  # origin th_sim
    "eval_interim": True,  # ?

    "feature_dim": 512,
    "is_norm_feat": True,
    "nhid": 1024,
    # "nlayer": 1,
    "nclass": 1,
    "dropout": 0.,
    "use_gcn_feat": True,
    # "max_conn": 10,
    "tau_0": 0.67,  # origin feature tau
    "tau": 0.67,  # gcn feature tau

    # spss parameter
    "M": 10,
    "N": 10,
    "K1_ratio": 0.8,
    "K2_ratio": 0.8,

    # inference parameters
    "threshold1": 0.3,
    "threshold2": 0.3,

    # optimizer
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-5,

    "save_decomposed_adj": False, # if True: sparse adjacency matrix ---> indices, values, shape
}