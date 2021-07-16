import os
import shutil

from FaceDet.FDAPI import FDAPI
from FaceQ.FQAPI import FQAPI
from FaceRec.FRAPI import FRAPI
from config import CONFIG

def test(cfg):

    FD = FDAPI(cfg)

    test_data_dir = "F:/FaceDRTools/test/test_data/"
    test_res_dir = "F:/FaceDRTools/test/test_res/"
    imgs = os.listdir(test_data_dir)
    for img in imgs:
        img_path = test_data_dir + img
        FD.save_align(img_path, test_res_dir)

    FQ = FQAPI(cfg)
    imgs = os.listdir(test_res_dir)
    q_res_dir = "F:/FaceDRTools/test/test_q/"
    for img in imgs:
        img_path = test_res_dir + img
        qs = FQ.get_q_score(img_path)
        shutil.copy(img_path, q_res_dir + "%.6f_%s" % (qs, img))

if __name__ == "__main__":
    cfg = CONFIG()
    test(cfg)