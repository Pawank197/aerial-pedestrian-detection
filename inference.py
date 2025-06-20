import os
import re
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models.retinanet import RetinaNet
from utils.transforms import Compose, ToTensor, Normalize

def find_latest_weights(ckpt_dir='checkpoints'):
    pattern = re.compile(r"weights_epoch_(\d+)\.pth$")
    max_epoch, latest_file = 0, None
    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch, latest_file = epoch, fname
    if not latest_file:
        raise FileNotFoundError(f"No weights_epoch_*.pth in {ckpt_dir}")
    return os.path.join(ckpt_dir, latest_file)

def inference(image_path, output_path='output.jpg', threshold=0.5):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select GPU if available[4]
    num_classes = 1

    # Load image
    image = Image.open(image_path).convert('RGB')  # open and convert image[5]
    transforms = Compose([
        ToTensor(),  # convert to tensor[6]
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # normalize[6]
    ])
    img_tensor, _ = transforms(image, {})  # apply transforms[7]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dimension[7]

    # Initialize model
    model = RetinaNet(num_classes=num_classes).to(device)  # create RetinaNet instance[8]

    # Load latest epoch weights
    weights_path = find_latest_weights('checkpoints')  # get latest weights file[9]
    model.load_state_dict(torch.load(weights_path, map_location=device))  # load weights[10]
    model.eval()

    # Run inference
    with torch.no_grad():
        boxes_list, scores_list, _ = model(img_tensor)  # forward pass[11]
    boxes, scores = boxes_list[0], scores_list[0]

    # Visualization
    plt.figure(figsize=(8,8))
    plt.imshow(image)  # show original image[12]
    ax = plt.gca()
    for box, score in zip(boxes, scores):  # loop through detections[13]
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)  # draw bounding box[14]
        ax.text(x1, y1 - 5, f'{score:.2f}', color='yellow',
                fontsize=12, weight='bold')  # annotate score[14]
    plt.axis('off')
    plt.savefig(output_path)  # save output image[15]
    print(f'Results saved to {output_path}')  # notify user[16]

if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]  # take image path from CLI[17]
    inference(img_path)
