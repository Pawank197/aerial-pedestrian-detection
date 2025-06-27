import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class AerialPedestrianDataset(Dataset):
    """
    Custom Dataset for aerial pedestrian detection with CSV annotations.
    """
    def __init__(self, annotations_file, labels_file, img_dir, transform=None):
        # Load annotations and class labels
        self.annotations = pd.read_csv(
            annotations_file,
            names=['image_path', 'x1', 'y1', 'x2', 'y2', 'class_name']
        )
        self.labels_df = pd.read_csv(
            labels_file,
            names=['class_name', 'class_id']
        )
        self.img_dir = img_dir
        self.transform = transform

        # Map class names to integer IDs
        self.class_to_id = dict(zip(
            self.labels_df['class_name'],
            self.labels_df['class_id']
        ))

        # Group annotations by image file
        self.grouped = self.annotations.groupby('image_path')
        self.image_paths = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Retrieve image path and load image
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Fetch annotations for this image
        records = self.grouped.get_group(img_name)
        boxes = records[['x1', 'y1', 'x2', 'y2']].values.astype(float).tolist()
        labels = records['class_name'].map(self.class_to_id).values.astype(int).tolist()

        if len(boxes) > 0:
            valid_boxes = []
            valid_labels = []
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                # Check for valid box dimensions
                if x2 > x1 and y2 > y1:  # Ensure positive width and height
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, image.shape[2] - 1))
                    y1 = max(0, min(y1, image.shape[1] - 1)) 
                    x2 = max(x1 + 1, min(x2, image.shape[2]))  # Ensure min width of 1
                    y2 = max(y1 + 1, min(y2, image.shape[1]))  # Ensure min height of 1
                    
                    valid_boxes.append([x1, y1, x2, y2])
                    valid_labels.append(label)
        
            boxes = valid_boxes
            labels = valid_labels

        # Convert to tensors
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        # Convert to tensors
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }

        return image, target
