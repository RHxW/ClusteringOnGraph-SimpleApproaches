import torch
import numpy as np
import torch.backends.cudnn as cudnn
import cv2

from FaceRec.model_irse import IR_SE_50


class FRAPI():
    def __init__(self, cfg):
        self.device = cfg.rec_device
        self.input_size = cfg.rec_size
        self.net = IR_SE_50(self.input_size).to(self.device)
        self.net.load_state_dict(torch.load(cfg.rec_checkpoint_path, map_location=self.device))
        self.net.eval()

        torch.set_grad_enabled(False)
        cudnn.benchmark = True

        self.RGB_MEAN = np.array([0.5507809, 0.4314252, 0.37640554], dtype=np.float32)
        self.RGB_STD = np.array([0.29048514, 0.25262007, 0.24208826], dtype=np.float32)

    def get_feature(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (np.array(img, dtype=np.float32) / 255 - self.RGB_MEAN) / self.RGB_STD
        img = img.transpose((2, 0, 1))
        input = torch.Tensor(img).to(self.device).unsqueeze(0)
        output = self.net(input).cpu()
        return output


