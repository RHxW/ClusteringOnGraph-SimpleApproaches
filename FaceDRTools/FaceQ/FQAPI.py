import torch
import numpy as np
import torch.backends.cudnn as cudnn
import cv2

from .faceQnet import faceQnet

class FQAPI():
    def __init__(self, cfg):
        self.device = cfg.q_device
        self.net = faceQnet().to(self.device)
        self.net.load_state_dict(torch.load(cfg.q_checkpoint_path, map_location=self.device))
        self.net.eval()

        torch.set_grad_enabled(False)
        cudnn.benchmark = True

        self.RGB_MEAN = np.array([0.5507809, 0.4314252, 0.37640554], dtype=np.float32)
        self.RGB_STD = np.array([0.29048514, 0.25262007, 0.24208826], dtype=np.float32)

    def get_q_score(self, img_path):
        image = cv2.imread(img_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = (np.array(img, dtype=np.float32) / 255 - self.RGB_MEAN) / self.RGB_STD
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(self.device)
        pred = self.net(img).squeeze(0).item()
        return pred
