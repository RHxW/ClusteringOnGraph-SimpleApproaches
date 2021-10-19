import os
from tqdm import tqdm
import datetime

from FaceDet.FDAPI import FDAPI
from FaceQ.FQAPI import FQAPI
from Config.config import CONFIG

def img_align_2_dir(cfg, src_root, dst_root):
    if not os.path.exists(src_root):
        return
    if src_root[-1] != "/":
        src_root += "/"
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)
    if dst_root[-1] != '/':
        dst_root += '/'

    FD = FDAPI(cfg)

    start_time = datetime.datetime.now()
    root_dirs = os.listdir(src_root)
    for sub_dir in tqdm(root_dirs):
        sd_path = src_root + sub_dir
        if not os.path.isdir(sd_path):
            continue
        sd_path += '/'
        dst_dir = dst_root + sub_dir + '/'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        imgs = os.listdir(sd_path)
        for name in imgs:
            img_path = sd_path + name
            if os.path.exists(dst_dir + name[:-4]+'.jpg'):
                continue
            FD.save_align(img_path, dst_dir)

    end_time = datetime.datetime.now()
    tc = end_time - start_time
    print("Total time consume: ", tc)


if __name__ == "__main__":
    cfg = CONFIG()
    src_root = ''
    dst_root = ''
    img_align_2_dir(cfg, src_root, dst_root)