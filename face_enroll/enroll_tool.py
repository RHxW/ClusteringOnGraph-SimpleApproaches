from face_enroll.enrollment import FaceEnrollmentINC
from face_enroll.enroll_utils import *
import random
import datetime


def enroll_tool_1dir(delta: int, res_name: str = ''):
    DB_root = "/home/songhui/COGSAs/face_enroll/test_res/"
    if not os.path.exists(DB_root):
        os.mkdir(DB_root)
    if res_name:
        res_name += '_'
    DB_root += res_name + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    os.mkdir(DB_root)
    enroll_API = FaceEnrollmentINC(tmp_DB_root=DB_root, clustering_method=1, get_id_face_method=3, q_th=0.1)
    # test_id_root = "/home/songhui/FaceClusterEnrollment/facepic_82id_align/"
    test_pic_root = "/home/songhui/wyh/face_data/yuemiao_211022/face_align/"  # 入库程序会把这个文件夹下的人脸图片删掉，需要额外准备一个临时文件夹
    tmp_root = "/home/songhui/COGSAs/face_enroll/enroll_tmp_data/"  # 将待入库图片复制到这个临时目录中
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    if tmp_root[-1] != '/':
        tmp_root += '/'
    dir_copy(test_pic_root, tmp_root)

    test_imgs = []
    for pname in os.listdir(tmp_root):
        test_imgs.append(tmp_root + pname)
    if delta <= 0:
        delta = len(test_imgs) + 1
    while test_imgs:
        _imgs = test_imgs[:delta]
        test_imgs = test_imgs[delta:]
        enroll_API.batch_enrollment(_imgs)

    singles = enroll_API.singles
    s_dir = DB_root + "-1/"
    os.mkdir(s_dir)
    for s_path in singles:
        s_name = s_path.split("/")[-1]
        shutil.copy(s_path, s_dir + s_name)

    # 删除空id文件夹
    res_ids = os.listdir(DB_root)
    for id_dir in res_ids:
        if len(os.listdir(DB_root + id_dir + "/")) == 0:
            shutil.rmtree(DB_root + id_dir + "/")

    # 删除临时保存的feature
    feature_root = enroll_API.feature_root
    if os.path.exists(feature_root):
        shutil.rmtree(feature_root)

    get_singularity_dirs(DB_root)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    enroll_tool_1dir(2000, 'yuemiao_211022')
    end_time = datetime.datetime.now()
    time_consume = end_time - start_time
    print("-" * 50)
    print("Time consume all: ", time_consume)
