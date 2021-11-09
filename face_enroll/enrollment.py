import os
import shutil
import torch
import numpy as np
import datetime

from vegcn.gcnv_api import GCNV_API
from vegcn.config.gcnv_config import CONFIG as gcnv_cfg
from FaceDRTools.FaceRec.FRAPI import FRAPI
from FaceDRTools.FaceQ.FQAPI import FQAPI
from FaceDRTools.Config.config import CONFIG
from face_enroll.enroll_utils import get_avg_feature_by_list, get_avg_feature, get_weight_feature_by_list


class FaceEnrollmentINC():
    def __init__(self, tmp_DB_root: str, clustering_method: int = 1, get_id_face_method: int = 2, q_th: float = 0.5):
        """

        :param tmp_DB_root: 保存结果id的dir，里面有-1（single），-2（低质量）和features文件夹
        :param clustering_method: 聚类方法：1. GCNV
        :param get_id_face_method: 获取id中心特征方法：1: 质量最高（测试效果最好）  2: 全部的平均特征  3: 高中低的平均特征  4: 高中低特征的加权平均
        """
        self.tmp_DB_root = tmp_DB_root  # 保存id的dir，里面有-1（single），-2（低质量）和features文件夹
        if not os.path.exists(self.tmp_DB_root):
            os.mkdir(self.tmp_DB_root)
        if self.tmp_DB_root[-1] != "/":
            self.tmp_DB_root += "/"

        if clustering_method not in [1, ]:
            raise RuntimeError("clustering_method error!")
        self.clustering_method = clustering_method  # 聚类方法：1. GCNV

        if get_id_face_method not in [1, 2, 3, 4]:
            raise RuntimeError("get_id_face_method error!")
        self.get_id_face_method = get_id_face_method  # 获取id中心特征方法：1: 质量最高（测试效果最好）  2: 全部的平均特征  3: 高中低的平均特征  4: 高中低特征的加权平均
        self.face_cfg = CONFIG()

        self.FRAPI = FRAPI(self.face_cfg)
        self.FQAPI = FQAPI(self.face_cfg)
        if q_th >= 1 or q_th <= 0:
            raise RuntimeError("Invalid quality threshold!")
        self.q_th = q_th

        # 入库图片按id保存feature（每种方案都保存特征）
        self.feature_root = self.tmp_DB_root + "features/"
        if self.get_id_face_method >= 1:
            os.mkdir(self.feature_root)

        # 临时存放单独图片（未入库成功的，-1类别）的特征
        self.singles_feature_dir = self.tmp_DB_root + "tmp_singles_feature/"
        if os.path.exists(self.singles_feature_dir):
            shutil.rmtree(self.singles_feature_dir)
        os.mkdir(self.singles_feature_dir)

        self.singles = []  # 未入库图片path列表

        self.ids = set()
        self.id_pics = dict()
        db_list = os.listdir(self.tmp_DB_root)
        if "features" in db_list:
            db_list.remove("features")
        if "tmp_singles_feature" in db_list:
            db_list.remove("tmp_singles_feature")
        for _id in db_list:
            self.ids.add(int(_id))
            self.id_pics[int(_id)] = []
            for _img in os.listdir(self.tmp_DB_root + _id + "/"):
                self.id_pics[int(_id)].append(self.tmp_DB_root + _id + "/" + _img)
        self.last_id = 1

        self.cluster_cfg = None
        if self.clustering_method == 1:
            self.cluster_cfg = gcnv_cfg
            self.cluster_API = GCNV_API(self.cluster_cfg)  # 调用`.cluster(features)`方法，传入的是np.array格式的特征矩阵
        else:
            raise RuntimeError("Cluster method not supported!")

    def batch_enrollment(self, img_paths: list):
        # 批量入库
        if type(img_paths) != list:
            raise RuntimeWarning("'img_paths' must be a list of paths")

        # 预先过滤低质量图片（之前是在self.get_cluster_result中）
        low_quality_dir = self.tmp_DB_root + "-2/"
        if not os.path.exists(low_quality_dir):
            os.mkdir(low_quality_dir)
        img_paths_quality_valid = []
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            q = self.FQAPI.get_q_score(img_path)
            if q < self.q_th:
                img_name = img_path.split("/")[-1]
                new_name = ("%.4f_" % q)[2:] + img_name
                shutil.move(img_path, low_quality_dir + new_name)
            else:
                img_paths_quality_valid.append(img_path)

        img_paths = list(set(self.singles + img_paths_quality_valid))
        # img_paths.sort()

        if len(img_paths) == 0:
            return

        start_time = datetime.datetime.now()

        N = len(img_paths)
        img_feats = []
        # singles保存特征文件，防止反复调用识别模型生成特征
        for i in range(len(self.singles)):
            _img_path = self.singles[i]
            _name = _img_path.split("/")[-1][:-4]
            _feat_path = self.singles_feature_dir + _name
            if os.path.exists(self.singles_feature_dir + _name):  # 读取特征
                _feat = np.fromfile(_feat_path, dtype=np.float32)
            else:  # 生成特征并保存
                _feat = self.FRAPI.get_feature(_img_path).numpy()
                _feat.tofile(_feat_path)
            img_feats.append(_feat)

        for i in range(len(self.singles), N):  # 对所有图片提特征
            img_path = img_paths[i]
            _feat = self.FRAPI.get_feature(img_path).numpy()
            img_feats.append(_feat)  # 特征

        ids, id_pic_path, id_pic_feat = self.get_ID_faces(self.get_id_face_method)
        total_ids = ids + [-1] * len(img_paths)  # 原始id，已入库的对应类别为id，未入库的对应类别为-1
        total_pic_paths = id_pic_path + img_paths  # 已入库id的path为"/xxx/0"，待入库图片的path为"/xxx/yyyy.jpg"
        total_pic_feats = id_pic_feat + img_feats  # 已有的id feature + 新入库的图片feature

        # feature_to_bin(self.bin_path, total_pic_feats)
        # get_list(self.list_path, total_pic_paths)
        total_pic_feats = np.array(total_pic_feats).reshape(-1, 512).astype("float32")
        cluster_id_res = self.cluster_API.cluster(total_pic_feats)

        # 入库
        single_idx = self.get_cluster_result(cluster_id_res, total_ids, total_pic_paths)
        self.singles = []
        for sid in single_idx:
            self.singles.append(total_pic_paths[sid])
        end_time = datetime.datetime.now()
        tc = end_time - start_time
        print("face_enroll time consume: ", tc)

    def get_ID_faces(self, method=1):
        # 获取每个id的人脸图片（用于聚类）
        # 多种方案
        # 1: 取质量分数top1
        # 2: 取（id下）当前全部图片的平均相似度，这种方式会保存特征（feature文件夹）
        # 3: 按质量分数的高中低三档取id下人脸，并求平均特征，这种方案会保存特征（feature文件夹）
        # 4: 按质量分数高中低加权（实际顺序为低中高）
        if method == 1:
            return self.get_ID_faces_1()
        elif method == 2:
            return self.get_ID_faces_2()
        elif method == 3:
            return self.get_ID_faces_3()
        elif method == 4:
            return self.get_ID_faces_4()
        else:
            return

    def get_ID_faces_1_(self):
        # 1: 取质量分数top1 （不保存特征文件）
        ids = list(self.id_pics.keys())
        id_pic_path = []
        id_pic_feat = []
        for _id in ids:  # 获取每个id下的质量最高者
            _pics = self.id_pics[_id]
            if not _pics:
                id_pic_path.append(None)
                id_pic_feat.append(None)
                self.id_pics.pop(_id)
                continue
            _pics.sort(reverse=True)
            id_pic_path.append(_pics[0])
            id_pic_feat.append(self.FRAPI.get_feature(_pics[0]).numpy())  # 每次都要重新提特征

        return ids, id_pic_path, id_pic_feat

    def get_ID_faces_1(self):
        # 1: 取质量分数top1
        # 修改为保存所有特征（feature文件夹）
        ids = list(self.id_pics.keys())
        id_pic_path = []
        id_pic_feat = []
        for _id in ids:
            _id_path = self.tmp_DB_root + str(_id) + "/"
            _pics = os.listdir(_id_path)
            _feat_path = self.feature_root + str(_id) + "/"
            if str(_id) not in os.listdir(self.feature_root):
                os.mkdir(_feat_path)
            _feats = os.listdir(_feat_path)
            if len(_pics) != len(_feats):
                # 找到未提特征的图片，将特征加入到features文件夹对应的id中
                for p in _pics:
                    pname = p.split(".")[0]
                    if pname in _feats:
                        continue
                    new_feature = self.FRAPI.get_feature(_id_path + p)
                    np.array(new_feature).tofile(_feat_path + pname)

            # _pics = self.id_pics[_id]
            feat_files = os.listdir(_feat_path)
            if not _pics:
                id_pic_path.append(None)
                id_pic_feat.append(None)
                self.id_pics.pop(_id)
                continue
            feat_files.sort(reverse=True)
            id_pic_path.append(feat_files[0])
            id_pic_feat.append(
                np.expand_dims(np.fromfile(_feat_path + feat_files[0], dtype=np.float32), 0))  # 每次都要重新提特征
        return ids, id_pic_path, id_pic_feat

    def get_ID_faces_2(self):
        # 2: 取（id下）当前全部图片的平均相似度，这种方案会保存特征（feature文件夹）
        ids = list(self.id_pics.keys())
        id_pic_path = []
        id_pic_feat = []
        for _id in ids:
            _id_path = self.tmp_DB_root + str(_id) + "/"
            _pics = os.listdir(_id_path)
            _feat_path = self.feature_root + str(_id) + "/"
            if str(_id) not in os.listdir(self.feature_root):
                os.mkdir(_feat_path)
            _feats = os.listdir(_feat_path)
            if len(_pics) != len(_feats):
                # 找到未提特征的图片，将特征加入到features文件夹对应的id中
                for p in _pics:
                    pname = p.split(".")[0]
                    if pname in _feats:
                        continue
                    new_feature = self.FRAPI.get_feature(_id_path + p)
                    np.array(new_feature).tofile(_feat_path + pname)
            feat = get_avg_feature(_feat_path)  # 取平均特征
            feat = np.expand_dims(feat, 0)  # (512) -> (1, 512)
            id_pic_path.append(_id_path + "0")
            id_pic_feat.append(feat)
        return ids, id_pic_path, id_pic_feat

    def get_ID_faces_3(self):
        # 3: 按质量分数的高中低三档取id下人脸，并求平均特征，这种方案会保存特征（feature文件夹）
        ids = list(self.id_pics.keys())
        id_pic_path = []
        id_pic_feat = []
        for _id in ids:
            _id_path = self.tmp_DB_root + str(_id) + "/"
            _pics = os.listdir(_id_path)
            _feat_path = self.feature_root + str(_id) + "/"
            if str(_id) not in os.listdir(self.feature_root):
                os.mkdir(_feat_path)
            _feats = os.listdir(_feat_path)
            if len(_pics) != len(_feats):
                # 找到未提特征的图片，将特征加入到features文件夹对应的id中
                for p in _pics:
                    pname = p.split(".")[0]
                    if pname in _feats:
                        continue
                    new_feature = self.FRAPI.get_feature(_id_path + p)
                    np.array(new_feature).tofile(_feat_path + pname)

            paths_all = os.listdir(_feat_path)
            paths_all.sort()
            _N = len(paths_all)
            if _N > 3:
                paths = []
                paths.append(paths_all[0])  # 质量排序第一个
                paths.append(paths_all[int(_N // 2)])  # 质量排序中间的
                paths.append(paths_all[-1])  # 质量排序最后一个
            else:
                paths = paths_all

            feat = get_avg_feature_by_list(_feat_path, paths)  # 取平均特征
            feat = np.expand_dims(feat, 0)  # (512) -> (1, 512)
            id_pic_path.append(_id_path + "0")
            id_pic_feat.append(feat)
        return ids, id_pic_path, id_pic_feat

    def get_ID_faces_4(self):
        # 4: 按质量分数取高中低，加权得到类中心特征，这种方案会保存特征（feature文件夹）
        ids = list(self.id_pics.keys())
        id_pic_path = []
        id_pic_feat = []
        weight = [0.2, 0.3, 0.5]  # 低中高的权重
        for _id in ids:
            _id_path = self.tmp_DB_root + str(_id) + "/"
            _pics = os.listdir(_id_path)
            _feat_path = self.feature_root + str(_id) + "/"
            if str(_id) not in os.listdir(self.feature_root):
                os.mkdir(_feat_path)
            _feats = os.listdir(_feat_path)
            if len(_pics) != len(_feats):
                # 找到未提特征的图片，将特征加入到features文件夹对应的id中
                for p in _pics:
                    pname = p.split(".")[0]
                    if pname in _feats:
                        continue
                    new_feature = self.FRAPI.get_feature(_id_path + p)
                    np.array(new_feature).tofile(_feat_path + pname)

            paths_all = os.listdir(_feat_path)
            paths_all.sort()
            _N = len(paths_all)
            if _N > 3:
                paths = []
                paths.append(paths_all[0])  # 质量排序第一个（最低）
                paths.append(paths_all[int(_N // 2)])  # 质量排序中间的
                paths.append(paths_all[-1])  # 质量排序最后一个（最高）
            else:
                paths = paths_all

            feat = get_weight_feature_by_list(_feat_path, paths, weight)  # 只有这个地方和`get_ID_faces_3`不一样
            feat = np.expand_dims(feat, 0)  # (512) -> (1, 512)
            id_pic_path.append(_id_path + "0")
            id_pic_feat.append(feat)
        return ids, id_pic_path, id_pic_feat

    def get_cluster_result(self, cluster_id_res, origin_ids, pic_paths):
        # 将聚类的结果合并/融合到已有库
        N = len(cluster_id_res)
        # remain_paths = []  # 未入库人脸图片path
        clusters = dict()  # 倒排索引  key: 伪标签（伪id）, value: [索引]
        singles = []  # 里面是单独一个类的item（图片）的索引，即未入库图片

        for i in range(N):
            o_id = origin_ids[i]
            c_id = cluster_id_res[i]
            if o_id == c_id == -1:  # 仍然是单独一张（未入库）
                singles.append(i)
            elif o_id != -1 and c_id == -1:  # 库中已经存在的id没变，而且没有新图聚到当前id下
                continue  # ??? TODO
            elif o_id == -1 and c_id != -1:  # 单独的图片聚类成功了！
                if c_id in clusters:
                    clusters[c_id].append(i)
                else:
                    clusters[c_id] = [i, ]
            elif o_id != -1 and c_id != -1:  # 库中已经存在的id与别的图片聚成一类，可能是新图入库，也可能是多个已有id聚成一类
                # 和上面那种情况一样，先写到clusters中
                if c_id in clusters:
                    clusters[c_id].append(i)
                else:
                    clusters[c_id] = [i, ]

        C_to_New = dict()  # id变更信息，包括：1. id间融合；2. 新图片入库；3. 创建新id
        # 通过倒排索引(clusters)进行查询
        for c_id in list(clusters.keys()):
            idx_list = clusters[c_id]
            o_id = -1
            for _idx in idx_list:
                if origin_ids[_idx] != -1:  # 遍历到的第一个id作为主id
                    o_id = origin_ids[_idx]
                    break
            if o_id == -1:  # 需要创建新id
                C_to_New[c_id] = self.last_id
                self.last_id += 1
            else:  # 把当前聚类所有图片合并到这个id下
                C_to_New[c_id] = o_id

        new_ids = dict()  # 合并/新增后的（待变更的！不变的不在里面）id信息，索引idx对应新id
        for i in range(N):
            c_id = cluster_id_res[i]
            if c_id == -1:  # 没变的不更新
                continue
            new_ids[i] = C_to_New[c_id]

        # 入库操作
        # 1. id合并
        # 2. 新图片入库（新建库与已有库）
        low_quality_dir = self.tmp_DB_root + "-2/"
        if not os.path.exists(low_quality_dir):
            os.mkdir(low_quality_dir)

        for idx in list(new_ids.keys()):
            n_id = new_ids[idx]
            o_id = origin_ids[idx]
            img_path = pic_paths[idx]
            if o_id == -1:  # 新图片入库
                q = self.FQAPI.get_q_score(img_path)  # 质量过滤（后修改为在self.batch_enrollment中进行过滤，即在入库前先进行过滤）
                img_name = img_path.split("/")[-1]
                new_name = ("%.4f_" % q)[2:] + img_name
                if q < self.q_th:
                    shutil.move(img_path, low_quality_dir + new_name)
                    continue

                if n_id in self.ids:  # 新图片入到已有库中
                    pass
                else:  # 新图片建库
                    os.mkdir(self.tmp_DB_root + "%d/" % n_id)
                    self.ids.add(n_id)
                    self.id_pics[n_id] = []
                new_path = self.tmp_DB_root + "%d/" % n_id + new_name
                shutil.copy(img_path, new_path)
                os.remove(img_path)
                self.id_pics[n_id].append(new_path)
            else:  # 已有id操作（并入别的id或不变）
                if o_id == n_id:  # 不变
                    continue
                else:  # 并入别的id
                    # 图片文件夹合并
                    from_dir = self.tmp_DB_root + "%d/" % o_id
                    des_dir = self.tmp_DB_root + "%d/" % n_id
                    for _img in os.listdir(from_dir):
                        o_path = from_dir + _img
                        n_path = des_dir + _img
                        shutil.move(o_path, n_path)
                        self.id_pics[n_id].append(n_path)
                    self.ids.remove(o_id)
                    self.id_pics.pop(o_id)

                    # 特征文件夹合并
                    if self.get_id_face_method >= 2:  # 使用2,3,4方式才会产生特征文件夹
                        feat_from_dir = self.feature_root + "%d/" % o_id
                        feat_des_dir = self.feature_root + "%d/" % n_id
                        for _feat in os.listdir(feat_from_dir):
                            o_path = feat_from_dir + _feat
                            n_path = feat_des_dir + _feat
                            shutil.move(o_path, n_path)

        return singles
