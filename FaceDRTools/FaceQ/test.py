import os
import shutil
from tqdm import tqdm
from FaceQ.FQAPI import FQAPI
from Config.config import CONFIG

def test_1dir_copy(src_root, dst_root, cfg):
    if src_root[-1] != '/':
        src_root += '/'
    assert os.path.exists(src_root)

    if dst_root[-1] != '/':
        dst_root += '/'
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    fqtool = FQAPI(cfg)

    imgs = os.listdir(src_root)
    for img in tqdm(imgs):
        img_path = src_root + img
        qscore = fqtool.get_q_score(img_path)
        new_name = "%.4f_" % qscore + img
        shutil.copyfile(img_path, dst_root + new_name)


if __name__ == '__main__':
    cfg = CONFIG()
    src_root = 'G:/20210624_yk/face_align/'
    dst_root = 'G:/20210624_yk/face_align_q/'
    test_1dir_copy(src_root, dst_root, cfg)