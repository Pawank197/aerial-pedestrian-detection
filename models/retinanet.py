import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models

from .fpn import FeaturePyramidNetwork
from .anchors import AnchorGenerator
from .losses import FocalLoss, SmoothL1Loss


class RetinaNet(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18', pretrained=True):
        super(RetinaNet, self).__init__()
        # Number of object classes (excluding background)
        self.num_classes = num_classes
        # Backbone: ResNet18 truncated at C5
        resnet = getattr(models, backbone_name)(pretrained=pretrained)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )  # outputs C3
        self.layer2 = resnet.layer3  # outputs C4
        self.layer3 = resnet.layer4  # outputs C5

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512], out_channels=128
        )
        # Heads
        self.cls_head = self._make_head(128, self.num_classes)
        self.reg_head = self._make_head(128, 4)
        # Anchors and losses
        self.anchor_gen = AnchorGenerator(
            sizes=[16, 32, 64, 128],
            aspect_ratios=[0.5, 1.0, 2.0]
        )
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1 = SmoothL1Loss(beta=1.0 / 9.0)

        # Initialize head biases
        prior_prob = 0.01
        bias_val = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        for layer in self.cls_head:
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, bias_val)

    def _make_head(self, in_channels, out_channels):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        return nn.Sequential(*layers)

    def _extract_features(self, x):
        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)
        return [c3, c4, c5]

    def forward(self, images, targets=None):
        # Features and FPN
        feats = self._extract_features(images)
        fpn_feats = self.fpn(OrderedDict(zip(['p3','p4','p5'], feats)))

        # Predictions
        cls_preds, reg_preds = [], []
        for feat in fpn_feats.values():
            cls_preds.append(self.cls_head(feat))
            reg_preds.append(self.reg_head(feat))

        # Training: compute loss
        if self.training and targets is not None:
            anchors = self.anchor_gen(images.shape[-2:], fpn_feats)
            return self.compute_loss(cls_preds, reg_preds, anchors, targets)
        # Inference: post-process
        return self.postprocess(cls_preds, reg_preds, fpn_feats, images.shape[-2:])

    def compute_loss(self, cls_preds, reg_preds, anchors, targets):
        cls_loss = self.focal_loss(cls_preds, anchors, targets)
        reg_loss = self.smooth_l1(reg_preds, anchors, targets)
        return cls_loss + reg_loss

    def postprocess(self, cls_preds, reg_preds, fpn_feats, image_shape):
        from torchvision.ops import nms
        device = cls_preds[0].device
        final_boxes, final_scores, final_labels = [], [], []
        # Loop through FPN levels
        anchors = self.anchor_gen(image_shape, fpn_feats)
        for cls_p, reg_p, anc in zip(cls_preds, reg_preds, anchors):
            B, C, H, W = cls_p.shape
            scores = cls_p.sigmoid().permute(0,2,3,1).reshape(B, -1, C)
            deltas = reg_p.permute(0,2,3,1).reshape(B, -1, 4)
            for b in range(B):
                sc, dt, an = scores[b], deltas[b], anc
                # Decode boxes
                widths  = an[:,2] - an[:,0]
                heights = an[:,3] - an[:,1]
                ctr_x   = an[:,0] + 0.5 * widths
                ctr_y   = an[:,1] + 0.5 * heights
                dx, dy, dw, dh = dt.unbind(1)
                pred_ctr_x = dx * widths + ctr_x
                pred_ctr_y = dy * heights + ctr_y
                pred_w = torch.exp(dw) * widths
                pred_h = torch.exp(dh) * heights
                boxes = torch.stack([
                    pred_ctr_x - 0.5*pred_w,
                    pred_ctr_y - 0.5*pred_h,
                    pred_ctr_x + 0.5*pred_w,
                    pred_ctr_y + 0.5*pred_h
                ], dim=1)
                # NMS per class
                for cls in range(1, self.num_classes+1):
                    cls_scores = sc[:,cls]
                    mask = cls_scores > 0.05
                    if mask.sum() == 0:
                        continue
                    bboxes = boxes[mask]
                    scores_f = cls_scores[mask]
                    keep = nms(bboxes, scores_f, 0.5)
                    final_boxes.append(bboxes[keep])
                    final_scores.append(scores_f[keep])
                    final_labels.extend([cls-1]*len(keep))
        return final_boxes, final_scores, final_labels
