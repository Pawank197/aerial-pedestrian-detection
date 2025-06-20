import torch
import itertools
import numpy as np


class AnchorGenerator:
    def __init__(self, sizes, aspect_ratios):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def __call__(self, image_shape, fpn_feats):
        """
        Returns list of anchors per feature map level.
        """
        device = list(fpn_feats.values())[0].device
        anchors_per_level = []
        img_h, img_w = image_shape
        for idx, feat in enumerate(fpn_feats.values()):
            _, _, H, W = feat.shape
            stride_h = img_h / H
            stride_w = img_w / W
            anchors = []
            for i, j in itertools.product(range(H), range(W)):
                cy = (i + 0.5) * stride_h
                cx = (j + 0.5) * stride_w
                for sz in self.sizes:
                    for ar in self.aspect_ratios:
                        w = sz * np.sqrt(ar)
                        h = sz / np.sqrt(ar)
                        anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            anchors = torch.tensor(anchors, device=device)
            anchors_per_level.append(anchors)
        return anchors_per_level
