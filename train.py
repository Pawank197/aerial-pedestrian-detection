import os
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
#   from models.retinanet import RetinaNet
## changes ##
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
## changes ##
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from collate_fn import pad_Collate

def find_latest_checkpoint(ckpt_dir):
    """
    Scan checkpoint files and return the highest epoch number.
    """
    pattern = re.compile(r"weights_epoch_(\d+)\.pth$")
    max_epoch = 0
    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            max_epoch = max(max_epoch, epoch)
    return max_epoch

def train():
    # Configuration
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select device[1]
    num_classes = 1 + 1                                                 # pedestrians[1] + background[0]
    batch_size  = 2
    num_epochs  = 15
    lr          = 1e-4

    # Data transforms
    train_transforms = Compose([
        ToTensor(),                                        # convert image to tensor[2]
        Normalize(mean=[0.485,0.456,0.406],                # normalize inputs[2]
                  std=[0.229,0.224,0.225]),
        RandomHorizontalFlip(prob=0.5)                     # augment data[3]
    ])

    # Dataset and DataLoader
    train_ds     = AerialPedestrianDataset(
                       'data/train_annotations.csv',
                       'data/labels.csv',
                       'data',
                       transform=train_transforms
                   )  # load train set[1]
    train_loader = DataLoader(
                       train_ds,
                       batch_size=batch_size,
                       shuffle=True,
                       collate_fn=pad_Collate
                   )  # batch loader[1]

    # Model, optimizer, scheduler

    # 1. Load a pretrained RetinaNet backbone (on COCO) and reset the head for your num_classes


    # Load model with pretrained weights
    model = retinanet_resnet50_fpn(
        weights=None,                            # no full‑model weights
        weights_backbone=ResNet50_Weights.DEFAULT,  # only the backbone
        num_classes=num_classes,                 # your 2 classes
        anchor_generator=AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
    )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)         # Adam optimizer[2]
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.1)         # LR scheduler[2]

    # Checkpoint directory setup
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)                        # ensure directory exists[4]    

    # Resume from latest checkpoint if available
    start_epoch = 0
    latest = find_latest_checkpoint(ckpt_dir)
    if latest > 0:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{latest}.pth')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # Adjust remaining epochs
    remaining_epochs = num_epochs - start_epoch

    # Training loop
    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (imgs, targets) in enumerate(train_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] | Step [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth'))
if __name__ == '__main__':
    train()
