import torch
import os
import cv2
import numpy as np
import shutil

from GenderClassifier.config import CONFIG
from GenderClassifier.model import GenderClsNetwork


def inference(cfg, data_root, res_root):
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    RGB_MEAN = np.array([0.5507809, 0.4314252, 0.37640554], dtype=np.float32)
    RGB_STD = np.array([0.29048514, 0.25262007, 0.24208826], dtype=np.float32)

    if not os.path.exists(data_root):
        return
    if data_root[-1] != "/":
        data_root += "/"
    if not os.path.exists(res_root):
        os.mkdir(res_root)
    if res_root[-1] != "/":
        res_root += "/"

    device = cfg["device"]

    network = GenderClsNetwork().to(device)
    network.eval()
    if os.path.exists(cfg["checkpoint_file"]):
        checkpoint = torch.load(cfg["checkpoint_file"], map_location=device)
        network.load_state_dict(checkpoint)
    else:
        return

    img_paths = []
    for img in os.listdir(data_root):
        img_paths.append(data_root + img)

    male_res = []
    female_res = []

    N = len(img_paths)

    for i in range(N):
        img = cv2.imread(img_paths[i])
        # img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (np.array(img, dtype=np.float32) / 255 - RGB_MEAN) / RGB_STD
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(device)
        pred = network(img).argmax().item()
        if pred == 1:
            male_res.append(img_paths[i])
        else:
            female_res.append(img_paths[i])

    male_dir = res_root + "male/"
    female_dir = res_root + "female/"
    for _dir in [male_dir, female_dir]:
        if os.path.exists(_dir):
            shutil.rmtree(_dir)
        os.mkdir(_dir)

    for img in male_res:
        name = img.split("/")[-1]
        shutil.copy(img, male_dir + name)
    for img in female_res:
        name = img.split("/")[-1]
        shutil.copy(img, female_dir + name)


if __name__ == "__main__":
    cfg = CONFIG
    data_root = "F:/gender_classifier/dataset/test_1/"
    res_root = "F:/gender_classifier/dataset/test_res/"
    inference(cfg, data_root, res_root)
