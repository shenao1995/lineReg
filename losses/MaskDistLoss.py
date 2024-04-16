import torch
import torch.nn as nn
from torchvision.ops import masks_to_boxes


class MaskDistanceLoss(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, pred_mask, target_mask):
        pred_mask = torch.argmax(pred_mask, dim=1)
        # target_mask = torch.argmax(target_mask, dim=1)
        pred_bbx = masks_to_boxes(pred_mask)
        target_bbx = masks_to_boxes(target_mask.squeeze(0))
        print(pred_bbx)
        print(target_bbx)
        pred_center_x = pred_bbx[0, 0] + (pred_bbx[0, 2] - pred_bbx[0, 0]) / 2
        pred_center_y = pred_bbx[0, 1] + (pred_bbx[0, 3] - pred_bbx[0, 1]) / 2
        target_center_x = target_bbx[0, 0] + (target_bbx[0, 2] - target_bbx[0, 0]) / 2
        target_center_y = target_bbx[0, 1] + (target_bbx[0, 3] - target_bbx[0, 1]) / 2
        x_l2 = torch.mean((pred_center_x - target_center_x)**2)
        y_l2 = torch.mean((pred_center_y - target_center_y)**2)
        x_loss = torch.mean(x_l2)
        y_loss = torch.mean(y_l2)
        return x_loss + y_loss


