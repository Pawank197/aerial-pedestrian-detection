import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_preds, anchors, targets):
        """
        cls_preds: list of tensors [B,C,H,W]; anchors: list; targets: list of dicts
        """
        losses = []
        for preds, anc, t in zip(cls_preds, anchors, targets):
            B, C, H, W = preds.shape
            pred = preds.permute(0,2,3,1).reshape(-1, C).sigmoid()
            gt_boxes = t['boxes']
            gt_labels = t['labels']
            # IoU matching (simplified)
            # build targets and compute focal loss
            # ...
            loss = F.binary_cross_entropy(pred, torch.zeros_like(pred))
            losses.append(loss)
        return torch.stack(losses).mean()


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0/9.0):
        super().__init__()
        self.beta = beta

    def forward(self, reg_preds, anchors, targets):
        losses = []
        for preds, anc, t in zip(reg_preds, anchors, targets):
            pred = preds.permute(0,2,3,1).reshape(-1, 4)
            gt = torch.zeros_like(pred)  # dummy encoding
            loss = F.smooth_l1_loss(pred, gt, beta=self.beta)
            losses.append(loss)
        return torch.stack(losses).mean()
