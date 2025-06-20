import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs  = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats: OrderedDict):
        names = list(feats.keys())
        xs = list(feats.values())
        # Top-down
        inner = self.lateral_convs[-1](xs[-1])
        outs = [self.output_convs[-1](inner)]
        for idx in range(len(xs)-2, -1, -1):
            lat = self.lateral_convs[idx](xs[idx])
            inner = lat + F.interpolate(inner, size=lat.shape[-2:], mode='nearest')
            outs.insert(0, self.output_convs[idx](inner))
        return OrderedDict(zip(names, outs))
