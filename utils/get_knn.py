import faiss
import nmslib
import multiprocessing
import numpy as np
from utils import Timer


def build_knns(feats, knn_method, k):
    with Timer('build knn'):
        if knn_method == "hnsw":
            knn = knn_hnsw(feats, k)
        elif knn_method == "faiss":
            knn = knn_faiss_cpu(feats, k)
        elif knn_method in ["faiss_gpu_single", "faiss_gpu"]:
            knn = knn_faiss_gpu_single(feats, k)
        elif knn_method == "faiss_gpu_all":
            knn = knn_faiss_gpu_all(feats, k)
        else:
            raise RuntimeError("knn_method invalid!")

    return knn


def knn_hnsw(feats, k, print_progress=True):
    index = nmslib.init(method='hnsw', space='cosinesimil')
    thread_count = multiprocessing.cpu_count()
    index.addDataPointBatch(feats)
    index.createIndex({
        'post': 2,
        'indexThreadQty': thread_count,
    },
        print_progress=print_progress)
    knns = index.knnQueryBatch(feats, k=k, num_threads=thread_count)
    return knns


def knn_faiss_cpu(feats, k):
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(feats)
    sims, nbrs = index.search(feats, k=k)
    knn = [(np.array(nbr, dtype=np.int32),
            np.array(1 - dist, dtype=np.float32))
           for nbr, dist in zip(nbrs, sims)]
    return knn


def knn_faiss_gpu_single(feats, k):
    d = feats.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(feats)
    sims, nbrs = index.search(feats, k=k)
    knn = [(np.array(nbr, dtype=np.int32),
            np.array(1 - dist, dtype=np.float32))
           for nbr, dist in zip(nbrs, sims)]
    return knn


def knn_faiss_gpu_all(feats, k):
    if faiss.get_num_gpus() < 1:
        print("faiss gpu not capable!!!")
        return
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(feats)
    sims, nbrs = index.search(feats, k=k)
    knn = [(np.array(nbr, dtype=np.int32),
            np.array(1 - dist, dtype=np.float32))
           for nbr, dist in zip(nbrs, sims)]
    return knn
