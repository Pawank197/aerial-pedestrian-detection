import torch
import torch.nn.functional as F

def pad_Collate(batch):
    # build a normal collate function, no padding or anyting
    return tuple(zip(*batch))
