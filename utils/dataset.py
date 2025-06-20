import os
import torch
from torch.utils.data import Dataset
import pandas as pd
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
        image = Image.open(img_path).convert('RGB')

        # Fetch annotations for this image
        records = self.grouped.get_group(img_name)
        boxes = records[['x1', 'y1', 'x2', 'y2']].values.astype(float)
        labels = records['class_name'].map(self.class_to_id).values.astype(int)

        # Convert to tensors
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        # Apply transforms (if any)
        if self.transform:
            image, target = self.transform(image, target)

        return image, target
