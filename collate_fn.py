import torch
import torch.nn.functional as F

def pad_collate(batch):
    images, targets = zip(*batch)
    # Find max H and W
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_imgs = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # pad = (left, right, top, bottom)
        padded = F.pad(img, (0, pad_w, 0, pad_h), value=0)  # zero-pad on right and bottom[3]
        padded_imgs.append(padded)

    batch_imgs = torch.stack(padded_imgs)  # now shapes match[1]
    return batch_imgs, targets
