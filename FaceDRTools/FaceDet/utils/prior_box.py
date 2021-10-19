import torch
from itertools import product as product
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']   # [[16, 32], [64, 128], [256, 512]],
        self.steps = cfg['steps']           # 特征图相对原图的缩小倍数
        self.clip = cfg['clip']
        self.image_size = image_size  # train:840*840
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        # 7.31 SH
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            step = self.steps[k]
            image_w = self.image_size[1]
            image_h = self.image_size[0]
            for i, j in product(range(f[0]), range(f[1])):  # 遍历特征图
                dense_x = (j + 0.5) * step / image_w  # +0.5是中心点，后续根据中心点与宽高转化成[x1, y1, x2, y2]
                dense_y = (i + 0.5) * step / image_h
                # 如果j*step/w 直接作为起始点，没有越界的anchors,  与(j - 0.5) 作为起始点（后续不做point_form） 是一样的
                for min_size in min_sizes:  # 一个特征点 对应len(min_sizes)个anchors, 尺寸为min_sizes
                    s_kx = min_size / image_w
                    s_ky = min_size / image_h

                    anchors += [dense_x, dense_y, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)  # 每个特征点对应两个anchor
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
