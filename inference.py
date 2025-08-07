import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model import YOLOv1
from dataset import YOLODataset
from utils import cellboxes_to_boxes, non_max_suppression, plot_image

class ComposeTransform:
    def __init__(self, resize=448):
        self.transforms = T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor(),
        ])

    def __call__(self, image, bboxes):
        image = self.transforms(image)
        return image, bboxes

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = ComposeTransform(resize=448)

    model = YOLOv1().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = YOLODataset("test.csv", "images", "labels", transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2
    )

    map_metric = MeanAveragePrecision()

    for idx, (x, label) in enumerate(test_loader):
        x = x.to(device)
        label = label.to(device)

        with torch.no_grad():
            predictions = model(x)

        pred_boxes = cellboxes_to_boxes(predictions)[0]
        pred_boxes = non_max_suppression(pred_boxes, iou_threshold=0.5, threshold=0.4)
        if idx < 5:
            img_np = x[0].permute(1, 2, 0).cpu()
            plot_image(img_np, pred_boxes)

        true_boxes = cellboxes_to_boxes(label)[0]

        if len(pred_boxes) > 0:
            preds_dict = {
                "boxes": torch.tensor([box[2:] for box in pred_boxes]),
                "scores": torch.tensor([box[1] for box in pred_boxes]),
                "labels": torch.tensor([int(box[0]) for box in pred_boxes]) 
            }
        else:
            preds_dict = {
                "boxes": torch.empty((0, 4)),
                "scores": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.int64)
            }

        if len(true_boxes) > 0:
            targets_dict = {
                "boxes": torch.tensor([box[2:] for box in true_boxes]),
                "labels": torch.tensor([int(box[0]) for box in true_boxes])
            }
        else:
            targets_dict = {
                "boxes": torch.empty((0, 4)),
                "labels": torch.tensor([], dtype=torch.int64)
            }

        map_metric.update([preds_dict], [targets_dict])

    results = map_metric.compute()
    print("Evaluation Results:")
    print(results)

if __name__ == '__main__':
    run_inference()