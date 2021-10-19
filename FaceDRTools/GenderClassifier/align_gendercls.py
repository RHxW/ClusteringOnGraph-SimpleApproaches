import os
import shutil
import torch
import numpy as np
import cv2
import tqdm

from FaceDet.FDAPI import FDAPI
from Config.config import CONFIG

from GenderClassifier.config import GC_CONFIG as GC_cfg
from GenderClassifier.model import GenderClsNetwork


def align_and_gendercls(API_cfg, GC_cfg, data_root, res_root, save_freq=10000):
    """
    保存两个txt文件(male_list, female_list)
    :param API_cfg:
    :param GC_cfg:
    :param data_root:
    :param res_root:
    :param save_freq:
    :return:
    """
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    RGB_MEAN = np.array([0.5507809, 0.4314252, 0.37640554], dtype=np.float32)
    RGB_STD = np.array([0.29048514, 0.25262007, 0.24208826], dtype=np.float32)

    FD = FDAPI(API_cfg)

    device = GC_cfg["device"]

    network = GenderClsNetwork().to(device)
    network.eval()
    if os.path.exists(GC_cfg["checkpoint_file"]):
        checkpoint = torch.load(GC_cfg["checkpoint_file"], map_location=device)
        network.load_state_dict(checkpoint)
    else:
        raise RuntimeError('Gender cls ckpt does not exist!!!')

    if not os.path.exists(data_root):
        return
    if data_root[-1] != "/":
        data_root += "/"

    if not os.path.exists(res_root):
        os.mkdir(res_root)
    if res_root[-1] != '/':
        res_root += '/'
    male_list_file = res_root + 'male_list.txt'
    female_list_file = res_root + 'female_list.txt'

    start_pos = 0
    if os.path.exists(male_list_file) and os.path.exists(female_list_file):
        with open(male_list_file, 'r') as f:
            start_pos += len(f.readlines())
        with open(female_list_file, 'r') as f:
            start_pos += len(f.readlines())

    print('#' * 50)
    print("Start @ position: %d!" % start_pos)
    print('#' * 50)

    male_imgs = []
    female_imgs = []

    total_imgs = os.listdir(data_root)
    for i in tqdm.tqdm(range(start_pos, len(total_imgs))):
        name = total_imgs[i]
        img_path = data_root + name
        if name.split(".")[-1] not in ['jpg', 'png']:
            os.remove(img_path)
            continue
        align_res = FD.get_align(img_path)
        if not align_res:
            if os.path.exists(img_path):
                os.remove(img_path)
            continue
        img_aligned = align_res[0]
        img = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        img = (np.array(img, dtype=np.float32) / 255 - RGB_MEAN) / RGB_STD
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(device)
        pred = network(img).argmax().item()
        if pred == 1:
            male_imgs.append(name + '\n')
        else:
            female_imgs.append(name + '\n')

        if (i + 1) % save_freq == 0:
            with open(male_list_file, 'a+') as f:
                f.writelines(male_imgs)
            with open(female_list_file, 'a+') as f:
                f.writelines(female_imgs)
            male_imgs = []
            female_imgs = []

    with open(male_list_file, 'a+') as f:
        f.writelines(male_imgs)
    with open(female_list_file, 'a+') as f:
        f.writelines(female_imgs)

    # deduplicate
    with open(male_list_file, 'r') as f:
        lns = f.readlines()
        res_set = set(lns)
    with open(male_list_file, 'w') as f:
        f.writelines(list(res_set))

    with open(female_list_file, 'r') as f:
        lns = f.readlines()
        res_set = set(lns)
    with open(female_list_file, 'w') as f:
        f.writelines(list(res_set))


if __name__ == "__main__":
    data_root = '/data/RAR_data/RAR_data/1/Images/'
    res_root = '/data/RAR_data/RAR_data/1/'
    API_cfg = CONFIG()
    save_freq = 2000
    align_and_gendercls(API_cfg, GC_cfg, data_root, res_root, save_freq)
