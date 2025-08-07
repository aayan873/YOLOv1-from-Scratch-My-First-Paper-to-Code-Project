import torch
import torchvision.transforms as T

def IOU(boxes_preds, boxes_labels):
    # Convert to tensors (if not already)
    if isinstance(boxes_preds, list):
        boxes_preds = torch.tensor(boxes_preds)
    if isinstance(boxes_labels, list):
        boxes_labels = torch.tensor(boxes_labels)

    # Ensure tensors are float
    boxes_preds = boxes_preds.float()
    boxes_labels = boxes_labels.float()

    # (x, y, w, h) to (x1, y1, x2, y2)
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Calculate intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


import torch

def cellboxes_to_boxes(predictions, S=7):

    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)
    
    bboxes_all = []

    for i in range(batch_size):
        bboxes = []
        pred = predictions[i]

        for row in range(S):
            for col in range(S):
                cell = pred[row, col]
                
                class_probs = cell[10:]
                class_pred = torch.argmax(class_probs)
                class_confidence = class_probs[class_pred]

                box1_conf = cell[0]
                box2_conf = cell[5]

                if box1_conf > box2_conf:
                    box = cell[1:5]
                    confidence = box1_conf
                else:
                    box = cell[6:10]
                    confidence = box2_conf

                x = (box[0] + col) / S
                y = (box[1] + row) / S
                w = box[2]
                h = box[3]

                if confidence > 0:
                    bboxes.append([
                        int(class_pred),
                        float(confidence),
                        float(x),
                        float(y),
                        float(w),
                        float(h),
                    ])

        bboxes_all.append(bboxes)

    return bboxes_all
def non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4, box_format="midpoint"):

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    nms_boxes = []

    while bboxes:
        best_box = bboxes.pop(0)
        nms_boxes.append(best_box)

        bboxes = [
            box for box in bboxes
            if box[0] != best_box[0] or IOU(best_box[2:], box[2:]) < iou_threshold
        ]

    return nms_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        class_label, conf, x, y, w, h = box
        x = x - w/2
        y = y - h/2
        rect = patches.Rectangle((x * image.shape[1], y * image.shape[0]), w * image.shape[1], h * image.shape[0], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x * image.shape[1], y * image.shape[0], f"{int(class_label)}: {conf:.2f}", color="white", bbox=dict(facecolor='red', alpha=0.5))
    plt.show()
