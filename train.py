import os
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.retinanet import RetinaNet
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

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
    num_classes = 1                                                         # pedestrians only[1]
    batch_size  = 4
    num_epochs  = 50
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
                       collate_fn=lambda x: tuple(zip(*x))
                   )  # batch loader[1]

    # Model, optimizer, scheduler
    model     = RetinaNet(num_classes=num_classes).to(device)  # initialize RetinaNet[4]
    optimizer = optim.Adam(model.parameters(), lr=lr)         # Adam optimizer[2]
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.1)         # LR scheduler[2]

    # Checkpoint directory setup
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)                        # ensure directory exists[4]

    # Resume from latest checkpoint if available
    start_epoch = 0
    latest_epoch = find_latest_checkpoint(ckpt_dir)
    if latest_epoch > 0:
        # Load epoch-specific weights and optimizer state
        weights_path = os.path.join(ckpt_dir, f'weights_epoch_{latest_epoch}.pth')
        opt_path     = os.path.join(ckpt_dir, 'latest.pth')
        checkpoint   = torch.load(opt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])              # load model state[1]
        optimizer.load_state_dict(checkpoint['optim'])          # load optimizer state[1]
        start_epoch = latest_epoch
        print(f"Resuming from epoch {start_epoch} of {num_epochs}")

    # Adjust remaining epochs
    remaining_epochs = num_epochs - start_epoch

    # Training loop
    for epoch_offset in range(1, remaining_epochs + 1):
        epoch = start_epoch + epoch_offset
        model.train()
        total_loss = 0.0

        for i, (imgs, targets) in enumerate(train_loader):
            imgs = torch.stack([img.to(device) for img in imgs])
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss = model(imgs, targets)                        # compute loss[4]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        scheduler.step()                                        # update lr[2]
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} Completed. Average Epoch Loss: {avg_loss:.4f}')

        # Save epoch-specific checkpoint
        epoch_path = os.path.join(ckpt_dir, f'weights_epoch_{epoch}.pth')
        torch.save(model.state_dict(), epoch_path)             # save distinct weights[1]

        # Update and save latest state (including optimizer)
        latest_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optimizer.state_dict()
        }
        torch.save(latest_state, os.path.join(ckpt_dir, 'latest.pth'))

if __name__ == '__main__':
    train()
