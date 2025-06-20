import torch
from torch.utils.data import DataLoader
from models.retinanet import RetinaNet
from utils.dataset import AerialPedestrianDataset
from utils.transforms import Compose, ToTensor, Normalize
from utils.evaluation import calculate_map

def evaluate():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select device[1]
    num_classes = 1
    batch_size = 4

    # Data transforms
    val_transforms = Compose([ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])  # validation transforms[2]

    # Dataset and loader
    val_ds = AerialPedestrianDataset('data/val_annotations.csv', 'data/labels.csv', 'data/val', transform=val_transforms)  # load val set[1]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))  # batch loader[1]

    # Model
    model = RetinaNet(num_classes=num_classes).to(device)  # init model[4]
    model.load_state_dict(torch.load('checkpoints/latest.pth', map_location=device)['model'])  # load checkpoint[1]
    model.eval()

    # Inference and collect predictions
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)  # inference mode returns detections[4]
            for out, tgt in zip(outputs, targets):
                boxes, scores, labels = out  # unpack output[4]
                all_preds.append({'boxes': boxes, 'scores': scores})
                all_targets.append({'boxes': tgt['boxes']})

    # Compute mAP
    mAP = calculate_map(all_preds, all_targets, iou_threshold=0.5)  # compute mAP[5]
    print(f'mAP@0.5: {mAP:.4f}')  # print result[5]

if __name__ == '__main__':
    evaluate()
