import torch
from torchvision.transforms import functional as F
import math

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

class RandomVerticalFlip:
    """
    Vertically flip the image and adjust bounding boxes with given probability.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            # Flip image
            image = image.flip(-2)  # Flip height dimension
            # Flip boxes
            h = image.shape[-2]
            boxes = target['boxes']
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]  # Flip y-coordinates
            target['boxes'] = boxes
        return image, target

class RandomRotation:
    """
    Rotate the image and adjust bounding boxes by a random angle.
    """
    def __init__(self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False):
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
        
    def __call__(self, image, target):
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        
        # Get image dimensions
        h, w = image.shape[-2:]
        center = (w / 2, h / 2)
        
        # Rotate image
        rotated_image = F.rotate(image, angle, self.interpolation, self.expand)
        
        # If expand is True, we need to calculate new dimensions
        if self.expand:
            # Calculate new dimensions after rotation
            angle_rad = math.radians(abs(angle))
            new_h = int(h * abs(math.cos(angle_rad)) + w * abs(math.sin(angle_rad)))
            new_w = int(h * abs(math.sin(angle_rad)) + w * abs(math.cos(angle_rad)))
            new_center = (new_w / 2, new_h / 2)
        else:
            new_h, new_w = h, w
            new_center = center
        
        # Rotate bounding boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            rotated_boxes = []
            
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # Get all four corners of the box
                corners = torch.tensor([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=torch.float)
                
                # Translate to origin
                corners -= torch.tensor([center[0], center[1]])
                
                # Rotate
                angle_rad = math.radians(angle)
                cos_val = math.cos(angle_rad)
                sin_val = math.sin(angle_rad)
                rotation_matrix = torch.tensor([
                    [cos_val, -sin_val],
                    [sin_val, cos_val]
                ], dtype=torch.float)
                rotated_corners = corners @ rotation_matrix.T
                
                # Translate back
                rotated_corners += torch.tensor([new_center[0], new_center[1]])
                
                # Get new bounding box from rotated corners
                x1_new = torch.min(rotated_corners[:, 0])
                y1_new = torch.min(rotated_corners[:, 1])
                x2_new = torch.max(rotated_corners[:, 0])
                y2_new = torch.max(rotated_corners[:, 1])
                
                rotated_boxes.append([x1_new, y1_new, x2_new, y2_new])
            
            target['boxes'] = torch.tensor(rotated_boxes, dtype=boxes.dtype)
        
        return rotated_image, target

class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, image, target):
        # Apply color jitter to image only, target remains unchanged
        brightness_factor = torch.empty(1).uniform_(max(0, 1 - self.brightness), 1 + self.brightness).item() if self.brightness > 0 else 1
        contrast_factor = torch.empty(1).uniform_(max(0, 1 - self.contrast), 1 + self.contrast).item() if self.contrast > 0 else 1
        saturation_factor = torch.empty(1).uniform_(max(0, 1 - self.saturation), 1 + self.saturation).item() if self.saturation > 0 else 1
        hue_factor = torch.empty(1).uniform_(-self.hue, self.hue).item() if self.hue > 0 else 0
        
        # Apply transforms in sequence
        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        image = F.adjust_saturation(image, saturation_factor)
        image = F.adjust_hue(image, hue_factor)
        
        return image, target

class RandomScale:
    """
    Randomly scale the image and adjust bounding boxes.
    """
    def __init__(self, scale_factor=0.2):
        self.scale_factor = scale_factor
        
    def __call__(self, image, target):
        # Randomly choose a scaling factor
        scale = torch.empty(1).uniform_(1 - self.scale_factor, 1 + self.scale_factor).item()
        
        # Get original dimensions
        h, w = image.shape[-2:]
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        resized_image = F.resize(image, [new_h, new_w])
        
        # Scale bounding boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes = boxes * scale
            target['boxes'] = boxes
        
        return resized_image, target
