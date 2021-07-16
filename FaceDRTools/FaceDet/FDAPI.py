import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import copy

from utils.model_loading import load_model
from FaceDet.retinaface import RetinaFace
from FaceDet.faceAlign import alignFace
from FaceDet.config import cfg_re50, cfg_mnet
from FaceDet.utils.py_cpu_nms import py_cpu_nms
from FaceDet.utils.box_utils import decode, decode_landm
from FaceDet.utils.prior_box import PriorBox
from FaceDet.utils.utils import scaledBox


class FDAPI():
    def __init__(self, cfg):
        self.cfg = cfg
        torch.set_grad_enabled(False)
        cudnn.benchmark = True

        if cfg.det_backbone == "mobile0.25":
            self.det_cfg = cfg_mnet
        elif cfg.det_backbone == "resnet":
            self.det_cfg = cfg_re50
        else:
            return

        self.device = cfg.det_device
        self.net = RetinaFace(cfg=self.det_cfg, phase="test").to(self.device)

        if not os.path.exists(cfg.det_checkpoint_path):
            raise RuntimeError("det checkpoint path does not exist!!!")
        self.net = load_model(self.net, cfg.det_checkpoint_path, self.device)
        self.net.eval()

        self.RGB_MEAN = np.array([0.5507809, 0.4314252, 0.37640554], dtype=np.float32)
        self.RGB_STD = np.array([0.29048514, 0.25262007, 0.24208826], dtype=np.float32)

    def img_padding(self, image):
        '''
        :param image:
        :return: padding图像， resize图像， resize比例
        '''
        # padding & resize to infer_size
        dst_w, dst_h = self.cfg.det_infer_size
        if dst_w == -1:
            return image, image, 1, [1, 1]

        img_h, img_w = image.shape[:2]
        max_side = max(img_w, img_h)

        ## 四周对称padding
        # start_w = (max_side - img_w) // 2
        # start_h = (max_side - img_h) // 2

        # 右下角padding
        start_w = 0
        start_h = 0

        if dst_w == dst_h:  # padding 成方形
            r_ratio = max_side / dst_w

            img_padding = np.zeros((max_side, max_side, 3), 'uint8')
            img_padding[start_h:start_h + img_h:, start_w:start_w + img_w] = image
            img_infer = cv2.resize(img_padding, (dst_w, dst_h), cv2.INTER_CUBIC)

        else:  # padding 成非方形
            if (img_w / img_h) < (dst_w / dst_h):  # 右侧补齐
                pad_w = img_h * dst_w // dst_h
                pad_h = img_h
            else:
                pad_w = img_w
                pad_h = img_w // (dst_w / dst_h)

            r_ratio = pad_w / dst_w
            img_padding = np.zeros((pad_h, pad_w, 3), 'uint8')
            img_padding[start_h:start_h + img_h:, start_w:start_w + img_w] = image
            img_infer = cv2.resize(img_padding, (dst_w, dst_h), cv2.INTER_CUBIC)

        return img_padding, img_infer, r_ratio, [start_w, start_h]

    def get_box(self, image, img_pad, img_infer, r_ratio):
        '''
        :param image: 原始图像， 用于裁剪检测图
        :param img_pad: pad图，用于1.5倍扩充
        :param img_infer: 128*128， 传入net
        :param r_ratio: resize比例，用于结果回退到原图
        :return: 检测图，以及对应的 检测坐标，关键点，置信度
        '''
        al_image_s = []
        kpts_s = []
        fbox_s = []
        text_s = []
        kpts_src = []

        img = np.float32(img_infer)
        im_height, im_width, _ = img.shape

        img -= self.cfg.rgb_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        scale = torch.Tensor([im_width, im_height] * 2)
        scale = scale.to(self.device)

        # img = torch.tensor(torch.ones([1, 3, 320, 320]))
        # img = img.cuda()

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.det_cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.squeeze(0), prior_data, self.det_cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).cpu().numpy()[:, 1]
        landms = decode_landm(landms.squeeze(0), prior_data, self.det_cfg['variance'])
        scale1 = torch.Tensor([im_width, im_height] * 5)
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.cfg.confidence_threshold)[0]
        if len(inds) == 0:
            return al_image_s, kpts_s, fbox_s, text_s, kpts_src
            # return 0, 0
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.cfg.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.cfg.nms_threshold)
        # keep = nms(dets, cfg_align['nms_threshold,force_cpu=cfg_align['cpu'])
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.cfg.keep_top_k, :]
        landms = landms[:self.cfg.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        for b in dets:
            if b[4] < self.cfg.vis_thres:
                continue
            text = "{:.4f}".format(b[4])

            # 回到原图（padding图）
            b = b * r_ratio

            kpts = np.array(b[5:15])
            kpts = kpts.reshape(5, 2).astype(np.float)

            b = list(map(int, b))

            fbox = scaledBox(b, img_pad, scale=self.cfg.box_scale)
            if self.cfg.use_box_scale:
                x1, y1, x2, y2 = fbox
                kpts -= [x1, y1]
                al_image = copy.deepcopy(img_pad[y1:y2, x1:x2, :])  # ~~~
                # if x2 - x1 < img_w // 2 or y2 - y1 < img_h // 2:
                #     continue

                if 0 in al_image.shape:
                    return al_image_s, kpts_s, fbox_s, text_s, kpts_src

                al_image_s.append(al_image)
                kpts_s.append(kpts)
                kpts_src.append(kpts + [x1, y1])
            else:
                al_image_s.append(image)
                kpts_s.append(kpts)
                kpts_src.append(kpts)

            fbox_s.append(fbox)
            text_s.append(text)

        return al_image_s, kpts_s, fbox_s, text_s, kpts_src

    def draw_box(self, image, kpts, box, txt, show_kpts=True):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(image, txt, (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0))
        if show_kpts:
            kpts = kpts.astype('int')
            for p1, p2 in kpts:
                cv2.circle(image, (int(p1), int(p2)), 1, (0, 255, 0), thickness=4)

        return image

    def detect(self, image):
        # 图像右下方用黑色padding，再resize 到infer_size
        img_pad, img_infer, infer_ratio, d_wh = self.img_padding(image)

        # 检测人脸
        img_det, kpts, box, text, kpts_src = self.get_box(image, img_pad, img_infer, infer_ratio)

        kpts_src = [x - d_wh for x in kpts_src]
        fbox = [[x[0] - d_wh[0], x[1] - d_wh[1], x[2] - d_wh[0], x[3] - d_wh[1]] for x in box]

        return img_det, kpts, fbox, text, kpts_src

    def save_det(self, img_path, save_dir, show_kpts=False):
        if not os.path.exists(img_path):
            return
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        det_res = self.detect(image)
        if not det_res:
            return
        img_det, kpts, fbox, text, kpts_src = det_res

        img_name = img_path.split("/")[-1][:-4]
        new_name = img_name + ".jpg"

        img_draw = copy.deepcopy(image)
        det_path = os.path.join(save_dir, new_name)
        for lms, box, txt in zip(kpts, fbox, text):
            img_draw = self.draw_box(img_draw, lms, box, txt, show_kpts=show_kpts)
        cv2.imwrite(det_path, img_draw)

    def save_align(self, img_path, save_dir):
        if not os.path.exists(img_path):
            return
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        det_res = self.detect(image)
        if not det_res:
            return
        img_det, kpts, fbox, text, kpts_src = det_res

        img_name = img_path.split("/")[-1][:-4]
        new_name = img_name + ".jpg"

        for ind, (img_d, lms) in enumerate(zip(img_det, kpts)):
            if ind > 0:
                continue
            _h, _w, _c = img_d.shape
            if min(_h, _w) <= 26:
                continue
            aligned_face = alignFace(img_d, lms, self.cfg.aligned_size, scale=self.cfg.aligned_scale)
            align_path = os.path.join(save_dir, new_name)
            cv2.imwrite(align_path, aligned_face)
