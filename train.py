import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.retinanet import RetinaNet
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

def train():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select device[1]
    num_classes = 1  # pedestrians only[1]
    batch_size = 4
    num_epochs = 50
    lr = 1e-4

    # Transforms
    train_transforms = Compose([
        ToTensor(),  # convert image to tensor[2]
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),  # normalize inputs[2]
        RandomHorizontalFlip(prob=0.5)  # augment data[3]
    ])

    # Datasets and loaders
    train_ds = AerialPedestrianDataset('data/train_annotations.csv', 'data/labels.csv', 'data/train', transform=train_transforms)  # load train set[1]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))  # batch loader[1]

    # Model, optimizer, scheduler
    model = RetinaNet(num_classes=num_classes).to(device)  # initialize RetinaNet[4]
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer[2]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # LR scheduler[2]

    # Checkpoints
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    ckpt_path = os.path.join(ckpt_dir, 'latest.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch'] + 1  # resume epoch[1]

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss = model(imgs, targets)  # compute loss[4]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()  # update lr[2]

        # Save checkpoint
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optim': optimizer.state_dict()
        }, ckpt_path)  # save latest[1]

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_{epoch+1}.pth'))  # periodic save[1]

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')  # log progress[2]

if __name__ == '__main__':
    train()
