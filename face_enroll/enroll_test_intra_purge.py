from face_enroll.enrollment import FaceEnrollmentINC
from face_enroll.enroll_utils import *
import random
import datetime
import time
from face_enroll.eval import fscore, get_label_eval
from face_enroll.intra_purge import enrollment_ids_intra_purge


def test(delta=500):
    DB_root = "/home/songhui/COGSAs/face_enroll/test_res/"
    if not os.path.exists(DB_root):
        os.mkdir(DB_root)
    DB_root += datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    os.mkdir(DB_root)
    enroll_API = FaceEnrollmentINC(tmp_DB_root=DB_root, clustering_method=1, get_id_face_method=1)
    # test_id_root = "/home/songhui/FaceClusterEnrollment/facepic_82id_align/"
    test_id_root = "/home/songhui/FaceClusterEnrollment/facepic_100115_align_526/"
    tmp_root = "/home/songhui/COGSAs/face_enroll/enroll_tmp_data/"  # 将待入库图片复制到这个临时目录中
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    dir_copy(test_id_root, tmp_root)
    test_imgs = get_pics_id_dir(tmp_root)
    random.shuffle(test_imgs)

    N = len(test_imgs) // delta + 1

    for i in range(N):
        _imgs = test_imgs[:delta]
        test_imgs = test_imgs[delta:]
        if i % 4 == 0:
            enrollment_ids_intra_purge(enroll_API.singles, enroll_API.tmp_DB_root)
        elif i == N - 1:
            enrollment_ids_intra_purge(enroll_API.singles, enroll_API.tmp_DB_root)
        enroll_API.batch_enrollment(_imgs)

    # while test_imgs:
    #     _imgs = test_imgs[:delta]
    #     test_imgs = test_imgs[delta:]
    #     enroll_API.batch_enrollment(_imgs)
    #     # time.sleep(random.randint(1, 2))

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

    get_singularity_dirs(DB_root)

    label_true, label_pred, valid, q_all = get_label_eval(test_id_root, DB_root)
    print('(singular removed) prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(
        *fscore(label_true[valid], label_pred[valid])))
    print(
        '(q_all) prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(*fscore(label_true[q_all], label_pred[q_all])))
    print("%d / %d / %d" % (len(label_true[valid]), len(label_true[q_all]), len(label_true)))


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    test(2000)
    end_time = datetime.datetime.now()
    time_consume = end_time - start_time
    print("-" * 50)
    print("Time consume all: ", time_consume)
