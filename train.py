import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from collate_fn import pad_Collate
from utils.evaluation import calculate_map
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator

# --- User-configurable checkpoint path ---
# Set this to your desired .pth file to resume training,
# or set to None to start from scratch:
checkpoint_path = 'checkpoints/101_checkpoint_epoch_6.pth'

# Training parameters
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2  # pedestrians + background
batch_size  = 2
num_epochs  = 15
lr          = 1e-4


# Data transforms
def get_transforms(is_train=True):
    transforms = [
        ToTensor(),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        RandomHorizontalFlip(prob=0.5),

    ]
    
    return Compose(transforms)


# Model creation
def create_model():
    # 1) Build a ResNet-101 + FPN backbone
    backbone = resnet_fpn_backbone(
        'resnet101',       # swap in ResNet-101 (instead of resnet50)
        pretrained=True,   # ImageNet weights
        trainable_layers=3 # how many top blocks to fine-tune
    )

    # 2) Plug it into the generic RetinaNet constructor
    model = RetinaNet(
        backbone,
        num_classes=num_classes,  # pedestrians + background
        anchor_generator=AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
    )

    return model.to(device)

# Training loop
def train():
    # Prepare data
    train_ds = AerialPedestrianDataset(
        'data/train_annotations.csv',
        'data/labels.csv',
        'data',
        transform=get_transforms()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_Collate)
    val_transforms = Compose([
        ToTensor(),                                        # convert image to tensor[2]
        Normalize(mean=[0.485,0.456,0.406],                # normalize inputs[2]
                  std=[0.229,0.224,0.225]),                
    ])

    # Dataset and DataLoader
    val_ds     = AerialPedestrianDataset(
                       'data/val_annotations.csv',
                       'data/labels.csv',
                       'data',
                       transform=val_transforms
                   )  # load train set[1]
    val_loader = DataLoader(
                       val_ds,
                       batch_size=batch_size,
                       shuffle=False,
                       collate_fn=pad_Collate
                   )  # batch loader[1]

    # Initialize model, optimizer, scheduler
    model     = create_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 0
    # Load checkpoint if specified
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt.get('optimizer', optimizer.state_dict()))
        scheduler.load_state_dict(ckpt.get('scheduler', scheduler.state_dict()))
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from checkpoint '{checkpoint_path}' at epoch {start_epoch}")

    # Actual training
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
                print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, save_path)
    
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

if __name__ == '__main__':
    train()