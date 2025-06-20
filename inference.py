import torch
from PIL import Image
import matplotlib.pyplot as plt
from models.retinanet import RetinaNet
from utils.transforms import Compose, ToTensor, Normalize

def inference(image_path, output_path='output.jpg', threshold=0.5):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select device[1]
    num_classes = 1

    # Load image and model
    image = Image.open(image_path).convert('RGB')  # load image[3]
    transforms = Compose([ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])  # preprocess[2]
    img_tensor, _ = transforms(image, {})  # get tensor[2]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # batch dimension[2]

    model = RetinaNet(num_classes=num_classes).to(device)  # init model[4]
    model.load_state_dict(torch.load('checkpoints/latest.pth', map_location=device)['model'])  # load weights[1]
    model.eval()

    # Inference
    with torch.no_grad():
        boxes_list, scores_list, labels_list = model(img_tensor)  # forward pass[4]
    boxes = boxes_list[0]
    scores = scores_list[0]

    # Visualization
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    ax = plt.gca()
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x1,y1,x2,y2 = box.cpu().numpy()
        rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.2f}', color='yellow', fontsize=12, weight='bold')
    plt.axis('off')
    plt.savefig(output_path)  # save inference image[3]
    print(f'Results saved to {output_path}')  # completion message[2]

if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]  # accept image path from CLI[2]
    inference(img_path)
