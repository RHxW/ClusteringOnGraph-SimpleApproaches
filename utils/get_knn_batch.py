import faiss
import nmslib
import multiprocessing
import numpy as np
from utils import Timer

class Build_kNN_Batch(object):
    def __init__(self, feats_all, knn_method):
        if knn_method == "hnsw":
            self.knn_constructor = kNN_HNSW(feats_all)
        elif knn_method == "faiss":
            self.knn_constructor = kNN_Faiss_CPU(feats_all)
        elif knn_method in ["faiss_gpu_single", "faiss_gpu"]:
            self.knn_constructor = kNN_Faiss_GPU_Single(feats_all)
        elif knn_method == "faiss_gpu_all":
            self.knn_constructor = kNN_Faiss_GPU_All(feats_all)
        else:
            raise RuntimeError("knn_method invalid!")

    def search(self, feats, k):
        return self.knn_constructor.search(feats, k)


class kNN_HNSW():
    def __init__(self, feats_all, print_progress=True):
        self.d = feats_all.shape[1]
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.thread_count = multiprocessing.cpu_count()
        self.index.addDataPointBatch(feats_all)
        self.index.createIndex({
        'post': 2,
        'indexThreadQty': self.thread_count,
        },
        print_progress=print_progress)

    def search(self, feats, k):
        assert feats.shape[1] == self.d, "feature dimension invalid, expect %d, but got %d" % (self.d, feats.shape[1])
        knns = self.index.knnQueryBatch(feats, k=k, num_threads=self.thread_count)
        return knns

class kNN_Faiss_CPU():
    def __init__(self, feats_all):
        self.d = feats_all.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(feats_all)

    def search(self, feats, k):
        assert feats.shape[1] == self.d, "feature dimension invalid, expect %d, but got %d" % (self.d, feats.shape[1])
        sims, nbrs = self.index.search(feats, k=k)
        knn = [(np.array(nbr, dtype=np.int32),
                np.array(1 - dist, dtype=np.float32))
               for nbr, dist in zip(nbrs, sims)]
        return knn

class kNN_Faiss_GPU_Single():
    def __init__(self, feats_all):
        if faiss.get_num_gpus() < 1:
            print("faiss gpu not capable!!!")
            return
        self.d = feats_all.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        self.index = faiss.GpuIndexFlatIP(res, self.d, flat_config)
        self.index.add(feats_all)

    def search(self, feats, k):
        assert feats.shape[1] == self.d, "feature dimension invalid, expect %d, but got %d" % (self.d, feats.shape[1])
        sims, nbrs = self.index.search(feats, k=k)
        knn = [(np.array(nbr, dtype=np.int32),
                np.array(1 - dist, dtype=np.float32))
               for nbr, dist in zip(nbrs, sims)]
        return knn

class kNN_Faiss_GPU_All():
    def __init__(self, feats_all):
        if faiss.get_num_gpus() < 1:
            print("faiss gpu not capable!!!")
            return
        self.d = feats_all.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(feats_all)

    def search(self, feats, k):
        assert feats.shape[1] == self.d, "feature dimension invalid, expect %d, but got %d" % (self.d, feats.shape[1])
        sims, nbrs = self.index.search(feats, k=k)
        knn = [(np.array(nbr, dtype=np.int32),
                np.array(1 - dist, dtype=np.float32))
               for nbr, dist in zip(nbrs, sims)]
        return knn