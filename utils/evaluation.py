import numpy as np

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two boxes.
    box format: [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea

    return interArea / unionArea if unionArea > 0 else 0.0

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) for single-class detection.
    predictions: list of dicts with 'boxes' and 'scores'
    targets: list of dicts with 'boxes'
    """
    all_scores = []
    all_matches = []
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        gt_boxes = gt['boxes'].cpu().numpy()
        total_gt += len(gt_boxes)

        matched = []
        for pb, score in zip(pred_boxes, pred_scores):
            best_iou = 0.0
            best_idx = -1
            for idx, gb in enumerate(gt_boxes):
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            match = 1 if best_iou >= iou_threshold and best_idx not in matched else 0
            if match:
                matched.append(best_idx)
            all_matches.append(match)
            all_scores.append(score)

    # Sort by score descending
    order = np.argsort(-np.array(all_scores))
    matches = np.array(all_matches)[order]
    tp = np.cumsum(matches)
    fp = np.cumsum(1 - matches)

    recall = tp / max(total_gt, np.finfo(np.float64).eps)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # 11-point interpolation for AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.any(recall >= t):
            ap += np.max(precision[recall >= t])
    ap /= 11.0

    return ap
