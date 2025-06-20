import torch
from torchvision.transforms import functional as F

class Compose:
    """
    Compose multiple image and target transforms sequentially.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    """
    Convert PIL image to a FloatTensor and normalize pixel values to [0,1].
    """
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize:
    """
    Normalize image tensor using mean and std per channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomHorizontalFlip:
    """
    Horizontally flip the image and adjust bounding boxes with given probability.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            # Flip image
            image = image.flip(-1)
            # Flip boxes
            w = image.shape[-1]
            boxes = target['boxes']
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes
        return image, target
