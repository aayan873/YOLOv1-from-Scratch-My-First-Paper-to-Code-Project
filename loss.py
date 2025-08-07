import torch
import torch.nn as nn
from utils import IOU

class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()

        self.S, self.B, self.C = S, B, C
        self.lambda_coord = 5
        self.lamda_noobj = 0.5
    
    def forward(self, predictions, target):
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        iou_b1 = IOU(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = IOU(predictions[..., 26:30], target[..., 21:25])

        iou_h, best_box = torch.max(torch.cat([iou_b1, iou_b2], dim=-1), dim=-1, keepdim=True)
        
        exists_box = target[..., 20].unsqueeze(3)
        
        # Localization Loss
        box1_c = predictions[..., 21:25]
        box2_c = predictions[..., 26:30]

        best_box = best_box.unsqueeze(-1)
        best_preds = best_box*box2_c + (1 - best_box)*box1_c
        box_targets = target[..., 21:25]

        best_preds[..., 2:4] = torch.sqrt(
            torch.abs(best_preds[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        loc_loss = nn.functional.mse_loss(
            exists_box * best_preds, exists_box * box_targets, reduction="sum"
        )

        # Confidence Loss
        conf1 = predictions[..., 20:21]
        conf2 = predictions[..., 25:26]
        best_conf = best_box * conf2 + (1 - best_box) * conf1

        obj_loss = nn.functional.mse_loss(
            exists_box * best_conf, exists_box * target[..., 20:21], reduction="sum"
        )

        noobj_loss = nn.functional.mse_loss(
            (1 - exists_box) * conf1, (1 - exists_box) * target[..., 20:21], reduction="sum"
        ) + nn.functional.mse_loss(
            (1 - exists_box) * conf2, (1 - exists_box) * target[..., 20:21], reduction="sum"
        )


        # Classification Loss

        pred_class = predictions[..., 0:20] 
        target_class = target[..., :self.C]
        class_loss = nn.functional.mse_loss(
            exists_box * pred_class, exists_box * target_class, reduction="sum"
        )

        loss = (
            self.lambda_coord * loc_loss
            + obj_loss
            + self.lamda_noobj * noobj_loss
            + class_loss
        )

        return loss








