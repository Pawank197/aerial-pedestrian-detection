import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.retinanet import RetinaNet
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize
from utils.evaluation import calculate_map
from collate_fn import pad_collate

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RetinaNet on test set for a specific epoch"
    )
    parser.add_argument(
        "--epoch", type=int, required=True,
        help="Epoch number whose weights to load (e.g., 10)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root directory for 'test' images and annotation CSVs"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default="checkpoints",
        help="Directory where epoch weight files are stored"
    )
    return parser.parse_args()

def test_eval(epoch, data_dir, ckpt_dir):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1
    batch_size = 4

    # Prepare transforms and dataset
    test_transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_ds = AerialPedestrianDataset(
        f"{data_dir}/test_annotations.csv",
        f"{data_dir}/labels.csv",
        f"{data_dir}",
        transform=test_transforms
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=pad_collate
    )

    # Initialize and load model weights for the specified epoch
    model = RetinaNet(num_classes=num_classes).to(device)
    ckpt_path = os.path.join(ckpt_dir, f"weights_epoch_{epoch}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Run inference on test set
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = torch.stack([img.to(device) for img in imgs])
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                boxes, scores, _ = out
                all_preds.append({"boxes": boxes, "scores": scores})
                all_targets.append({"boxes": tgt["boxes"]})

    # Compute and print mAP@0.5 for the specified epoch
    mAP = calculate_map(all_preds, all_targets, iou_threshold=0.5)
    print(f"Epoch {epoch} — Test mAP@0.5: {mAP:.4f}")

if __name__ == "__main__":
    args = parse_args()
    test_eval(args.epoch, args.data_dir, args.ckpt_dir)
