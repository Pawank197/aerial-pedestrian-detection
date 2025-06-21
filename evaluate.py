import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize
from utils.evaluation import calculate_map
from collate_fn import pad_collate

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RetinaNet on validation set for a specific epoch"
    )
    parser.add_argument(
        "--epoch", type=int, required=True,
        help="Epoch number whose weights to load (e.g., 10)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root directory for 'val' images and annotation CSVs"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default="checkpoints",
        help="Directory where epoch checkpoint files are stored"
    )
    return parser.parse_args()

def evaluate(epoch, data_dir, ckpt_dir):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # pedestrians + background
    batch_size = 4

    # Prepare transforms and dataset
    val_transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_ds = AerialPedestrianDataset(
        f"{data_dir}/val_annotations.csv",
        f"{data_dir}/labels.csv",
        f"{data_dir}",
        transform=val_transforms
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=pad_collate
    )

    # Initialize model with same config as training
    model = retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        anchor_generator=AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    # Run inference on validation set
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                # out is a dict with keys 'boxes', 'labels', 'scores'
                all_preds.append({
                    'boxes': out['boxes'].cpu(),
                    'scores': out['scores'].cpu(),
                    'labels': out['labels'].cpu()
                })
                all_targets.append({
                    'boxes': tgt['boxes'],
                    'labels': tgt['labels'] if 'labels' in tgt else None
                })

    # Compute and print mAP@0.5 for the specified epoch
    mAP = calculate_map(all_preds, all_targets, iou_threshold=0.5)
    print(f"Epoch {epoch} — mAP@0.5: {mAP:.4f}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.epoch, args.data_dir, args.ckpt_dir)
