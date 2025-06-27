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
from utils.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomScale, RandomRotation
from collate_fn import pad_Collate
from utils.evaluation import calculate_map
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.optim.lr_scheduler import LinearLR, SequentialLR

checkpoint_path = None

# Training parameters
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2  # pedestrians + background
batch_size  = 8
num_epochs  = 25
lr          = 1e-3
g = torch.Generator()
g.manual_seed(42)

# Data transforms
def train_transform():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomScale(scale_factor=(0.8, 1.2)),
        RandomRotation(degrees=10)
    ])

def val_transform():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Model creation
def create_model():
    # 1) Build a ResNet-101 + FPN backbone
    backbone = resnet_fpn_backbone(
        'resnet101',       # swap in ResNet-101 (instead of resnet50)
        pretrained=True,   # ImageNet weights
        trainable_layers=3, # how many top blocks to fine-tune
    )

    # 2) Plug it into the generic RetinaNet constructor
    model = RetinaNet(
        backbone,
        num_classes=num_classes,  # pedestrians + background
        anchor_generator=AnchorGenerator(
            sizes=((16, 22, 32), (32, 45, 64), (64, 90, 128), (128, 180, 256), (256, 362, 512)),
            aspect_ratios=((0.5, 1.0, 2.0, 3.0),) * 5,   # Added 3.0 ratio for pedestrians
            ),
            focal_loss_alpha=0.25,  # Default but explicit
            focal_loss_gamma=2.0    # Proven optimal for aerial imagery
        )

    return model.to(device)

# Training loop
def train():
    # Prepare data
    train_ds = AerialPedestrianDataset(
        'data/train_annotations.csv',
        'data/labels.csv',
        'data',
        transform=train_transform()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_Collate, generator=g)

    # Dataset and DataLoader
    val_ds     = AerialPedestrianDataset(
                       'data/val_annotations.csv',
                       'data/labels.csv',
                       'data',
                       transform=val_transform()
                   )  # load train set[1]
    val_loader = DataLoader(
                       val_ds,
                       batch_size=batch_size,
                       shuffle=False,
                       collate_fn=pad_Collate
                   )  # batch loader[1]

    # Initialize model, optimizer, scheduler
    model     = create_model()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr,             # Initial learning rate
        weight_decay=1e-4, # weight decay
        eps=1e-8
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=1000  # 1000 iterations warmup
    )

    main_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 20],  # Later milestones
        gamma=0.1
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[1000]
    )

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