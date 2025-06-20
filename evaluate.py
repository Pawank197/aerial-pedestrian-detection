import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.retinanet import RetinaNet
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize
from utils.evaluation import calculate_map

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RetinaNet on validation set for a specific epoch"
    )  # use argparse for CLI parsing[2]
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
        help="Directory where epoch weight files are stored"
    )
    return parser.parse_args()

def evaluate(epoch, data_dir, ckpt_dir):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select GPU if available[1]
    num_classes = 1
    batch_size = 4

    # Prepare transforms and dataset
    val_transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  # validation image preprocessing[2]
    val_ds = AerialPedestrianDataset(
        f"{data_dir}/val_annotations.csv",
        f"{data_dir}/labels.csv",
        f"{data_dir}",
        transform=val_transforms
    )  # load validation set[3]
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )  # batch loader for validation[3]

    # Initialize and load model weights for the specified epoch
    model = RetinaNet(num_classes=num_classes).to(device)  # instantiate RetinaNet[1]
    ckpt_path = os.path.join(ckpt_dir, f"weights_epoch_{epoch}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))  # load epoch-specific weights[1]
    model.eval()

    # Run inference on validation set
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = torch.stack([img.to(device) for img in imgs])
            outputs = model(imgs)  # get detections for batch[4]
            for out, tgt in zip(outputs, targets):
                boxes, scores, _ = out
                all_preds.append({"boxes": boxes, "scores": scores})
                all_targets.append({"boxes": tgt["boxes"]})

    # Compute and print mAP@0.5 for the specified epoch
    mAP = calculate_map(all_preds, all_targets, iou_threshold=0.5)  # compute mAP metric[3]
    print(f"Epoch {epoch} — mAP@0.5: {mAP:.4f}")  # display result[3]

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.epoch, args.data_dir, args.ckpt_dir)
